import numpy as np
import pinocchio as pin
import proxsuite
from scipy.linalg import lstsq


# =============================================================================
# 1. WholeBodyTorqueGenerator  (from whole_body_torque.cpp)
#    G z = f  →  z* = (G^T G + W)^{-1} G^T f  →  tau = z*[nv:]
# =============================================================================

class WholeBodyTorqueGenerator:
    """
    Floating-base dynamics + contact constraint 연립으로 tau를 직접 계산.

    G = [M(q)   -S^T ]    f = [ -nle + J_c^T F_hat ]
        [J_c(q)   0  ]        [ -Jdot_qdot          ]

    z = [ddq(nv), tau(na)],  z* = argmin ||G z - f||² + z^T W z

    NOTE: G^TG 정규방정식은 조건수가 G의 제곱이 되어 수치적으로 위험.
          대신 augmented LS를 SVD(gelsd)로 직접 풀이:
              [ G    ] z = [ f ]
              [ √W   ]     [ 0 ]
    """

    def __init__(self, nv: int, na: int, nc: int,
                 w_ddq: float = 1e-4, w_tau: float = 1e-3):
        self.nv, self.na, self.nc = nv, na, nc
        self.n_z = nv + na

        self.G = np.zeros((nv + nc, self.n_z))

        # -S^T: floating base(6) = 0, actuated(na) = -I  (고정)
        self.G[6:nv, nv:] = -np.eye(na)

        # Tikhonov regularization → augmented row √W
        sqrtW = np.zeros((self.n_z, self.n_z))
        np.fill_diagonal(sqrtW[:nv, :nv], np.sqrt(w_ddq))
        np.fill_diagonal(sqrtW[nv:, nv:], np.sqrt(w_tau))

        # 사전 할당: A = [G; √W], b = [f; 0]
        self._A = np.zeros((nv + nc + self.n_z, self.n_z))
        self._A[nv + nc:, :] = sqrtW
        self._b = np.zeros(nv + nc + self.n_z)

        self.J_c = np.zeros((nc, nv))
        self.Jdot_qdot = np.zeros(nc)

    def compute(self, pw, rf_key: str, lf_key: str, F_hat: np.ndarray) -> np.ndarray:
        """
        pw      : Pinocchio_Wrapper (update_model 완료 상태)
        rf_key  : fid dict key (e.g. "R_foot")
        lf_key  : fid dict key (e.g. "L_foot")
        F_hat   : (nc,) optimal contact force
        returns : tau (na,)
        """
        model, data = pw.model, pw.data
        rf_id = pw.fid[rf_key]
        lf_id = pw.fid[lf_key]

        # contact Jacobian (update_model에서 이미 jacobians 계산됨)
        self.J_c[:6] = pin.getFrameJacobian(model, data, rf_id, pin.LOCAL_WORLD_ALIGNED)
        self.J_c[6:] = pin.getFrameJacobian(model, data, lf_id, pin.LOCAL_WORLD_ALIGNED)

        # Jdot*qdot (발이 땅에 고정 → desired classical acc = 0)
        self.Jdot_qdot[:6] = pin.getFrameClassicalAcceleration(
            model, data, rf_id, pin.LOCAL_WORLD_ALIGNED).vector
        self.Jdot_qdot[6:] = pin.getFrameClassicalAcceleration(
            model, data, lf_id, pin.LOCAL_WORLD_ALIGNED).vector

        # wrapper view 직접 사용 (no copy)
        M, _, nle = pw.compute_dynamics_term()

        # G 블록 갱신
        nv, nc = self.nv, self.nc
        self.G[:nv, :nv]   = M
        self.G[nv:, :nv]   = self.J_c

        # f 갱신
        self._b[:nv]      = -nle
        self._b[:nv]     += self.J_c.T @ F_hat
        self._b[nv:nv+nc] = -self.Jdot_qdot
        # _b[nv+nc:] = 0 (sqrtW에 곱해지는 RHS, 항상 0)

        # augmented LS (조건수 안전)
        self._A[:nv+nc, :] = self.G
        z, *_ = lstsq(self._A, self._b, lapack_driver='gelsd', check_finite=False)

        return z[nv:]   # tau (na,)


# =============================================================================
# 2. CoMDynamics  (from com_dynamics.cpp)
#    K F = u,  K(6×12), F(12,) = [f_R(3), tau_R(3), f_L(3), tau_L(3)]
# =============================================================================

class CoMDynamics:
    """
    Linear/angular momentum dynamics → K matrix + u vector for Force QP.

    [ D1 ] F = [ m*ddx_des + F_g ]
    [ D2 ]     [ dL               ]
    """

    def __init__(self, mass: float, gravity: float = 9.81):
        self.m = mass
        self.g = gravity
        self.K = np.zeros((6, 12))
        self.u = np.zeros(6)

        # D1 = [I 0 I 0]  (선형 운동량, 고정)
        self._D1 = np.zeros((3, 12))
        self._D1[:, 0:3] = np.eye(3)   # f_R
        self._D1[:, 6:9] = np.eye(3)   # f_L

    def update(self, pw, ddc_des: np.ndarray, dL: np.ndarray = None):
        """
        pw      : Pinocchio_Wrapper (update_model 완료 상태)
        ddc_des : (3,) desired CoM acceleration
        dL      : (3,) rate of angular momentum (default 0)
        """
        if dL is None:
            dL = np.zeros(3)

        com_pos = pw.pos_com_world
        r_R = pw.oM_Rfoot.translation - com_pos
        r_L = pw.oM_Lfoot.translation - com_pos

        # K = [ I    0   I    0  ]
        #     [ [r_R]× I [r_L]× I ]
        self.K[:3] = self._D1
        self.K[3:6, 0:3]  = pin.skew(r_R)
        self.K[3:6, 3:6]  = np.eye(3)
        self.K[3:6, 6:9]  = pin.skew(r_L)
        self.K[3:6, 9:12] = np.eye(3)

        self.u[0]  = self.m * ddc_des[0]
        self.u[1]  = self.m * ddc_des[1]
        self.u[2]  = self.m * ddc_des[2] + self.m * self.g
        self.u[3:] = dL


class ForceOptimizerProx:
    """
    ProxQP 기반 접촉력 최적화. OSQP와 동일한 문제/제약 구조.
    F (12,) = [f_R(3), tau_R(3), f_L(3), tau_L(3)]

    __init__에서 행렬 크기를 고정해 메모리를 미리 할당하고,
    solve() 호출마다 update()로 데이터만 교체 → warm-start 유지.
    """

    N_VARS = 12
    N_EQ   = 0
    N_INEQ = 10   # 마찰원뿔 5행 × 2발

    def __init__(self, mu: float = 0.5, fz_max: float = 500.0):
        self.mu = mu
        self.fz_max = fz_max
        self.opt_F = np.zeros(self.N_VARS)

        # 마찰원뿔 constraint 행렬 (고정, 매 스텝 불변)
        self._A, self._l, self._u = self._build_constraints()

        # ProxQP: 크기를 명시해 메모리 사전 할당
        self._qp = proxsuite.proxqp.dense.QP(
            self.N_VARS, self.N_EQ, self.N_INEQ
        )
        self._qp.settings.eps_abs     = 1e-4
        self._qp.settings.eps_rel     = 1e-4
        self._qp.settings.max_iter    = 1000
        self._qp.settings.verbose     = False
        self._qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT

        # 더미 H, g로 init (실제 값은 첫 solve에서 설정)
        H0 = np.eye(self.N_VARS)
        g0 = np.zeros(self.N_VARS)
        self._qp.init(H0, g0,
                      None, None,          # equality: 없음
                      self._A, self._l, self._u)
        self._initialized = False

        # 사전 할당 버퍼 (solve()에서 재사용, 매 호출 alloc 회피)
        self._W_reg = 1e-4 * np.eye(self.N_VARS)
        self._H_buf = np.empty((self.N_VARS, self.N_VARS))
        self._g_buf = np.empty(self.N_VARS)

    def _build_constraints(self):
        mu_eff = self.mu / np.sqrt(2.0)
        INF = 1e9
        A3 = np.array([[ 1,  0, -mu_eff],
                       [-1,  0, -mu_eff],
                       [ 0,  1, -mu_eff],
                       [ 0, -1, -mu_eff],
                       [ 0,  0,  1     ]], dtype=np.float64)
        l3 = np.array([-INF, -INF, -INF, -INF, 0.0])
        u3 = np.array([ 0.0,  0.0,  0.0,  0.0, self.fz_max])

        A6 = np.zeros((5, 6))
        A6[:, :3] = A3

        A = np.zeros((self.N_INEQ, self.N_VARS))
        A[:5,  0:6]  = A6
        A[5:, 6:12]  = A6
        return A, np.concatenate([l3, l3]), np.concatenate([u3, u3])

    def solve(self, K: np.ndarray, u_vec: np.ndarray,
              W_reg: np.ndarray = None) -> np.ndarray:
        """
        K     : (6, 12)
        u_vec : (6,)
        W_reg : (12, 12) optional. None이면 init에서 만든 _W_reg 사용.
        returns opt_F (12,)
        """
        Wreg = self._W_reg if W_reg is None else W_reg

        # H = 2*(K^T K + W_reg) — 사전 할당 버퍼에 in-place 누적
        np.dot(K.T, K, out=self._H_buf)
        self._H_buf += Wreg
        self._H_buf *= 2.0

        # g = -2 * K^T u_vec
        np.dot(K.T, u_vec, out=self._g_buf)
        self._g_buf *= -2.0

        if not self._initialized:
            self._qp.init(self._H_buf, self._g_buf,
                          None, None, self._A, self._l, self._u)
            self._initialized = True
        else:
            self._qp.update(H=self._H_buf, g=self._g_buf)

        self._qp.solve()

        if self._qp.results.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED:
            self.opt_F = np.array(self._qp.results.x)

        return self.opt_F


# =============================================================================
# 4. WholeBodyController  (DBFC_core 흐름)
#    BalanceTask → CoMDynamics → ForceOptimizer → WholeBodyTorqueGenerator
# =============================================================================

class WholeBodyController:
    """
    DBFC 흐름:
      1. BalanceTask (PD)     : com_des → ddc_des
      2. CoMDynamics          : ddc_des → K, u
      3. ForceOptimizer (QP)  : K, u → F_hat
      4. WholeBodyTorqueGenerator : F_hat → tau

    외부에서 pw.update_model(q, dq) 호출 후 compute() 사용.
    """

    def __init__(self, pw,
                 # ── 자주 튜닝하는 게인 (위로 노출) ──
                 kp_com: float = 100.0,
                 kd_com: float = 20.0,
                 mu: float = 0.5,
                 # ── 모델 차원 / 셋업 ──
                 nc: int = 12,
                 contact_keys=None,
                 # ── 동역학 LS regularization ──
                 w_ddq: float = 1e-4,
                 w_tau: float = 1e-3):
        """
        pw            : Pinocchio_Wrapper
        kp_com, kd_com: BalanceTask PD 게인 (워킹 시 30/10 정도로 낮추는 게 보통)
        mu            : friction coefficient
        nc            : contact DOF (6 per foot × 2 = 12)
        contact_keys  : pw.fid 키 리스트. 기본 ["R_foot", "L_foot"].
        w_ddq, w_tau  : Tikhonov regularization (작을수록 dynamics-strict)
        """
        self.pw = pw
        self.kp_com = kp_com
        self.kd_com = kd_com
        self.contact_keys = list(contact_keys) if contact_keys else ["R_foot", "L_foot"]

        self.torque_gen = WholeBodyTorqueGenerator(pw.nv, pw.na, nc,
                                                   w_ddq=w_ddq, w_tau=w_tau)
        self.com_dyn    = CoMDynamics(pw.mass)
        self.force_opt  = ForceOptimizerProx(mu=mu)

    def compute(self, com_des: np.ndarray, com_dot_des: np.ndarray,
                dL_des: np.ndarray = None) -> np.ndarray:
        """
        com_des     : (3,) desired CoM position  [world frame]
        com_dot_des : (3,) desired CoM velocity  [world frame]
        dL_des      : (3,) desired angular momentum rate (default 0 — LIPM 가정)
        returns     : tau (na,)

        전제: pw.update_model(q, dq) 가 이미 호출된 상태.
        """
        pw = self.pw

        # 1. BalanceTask: PD → ddc_des
        ddc_des = (self.kp_com * (com_des     - pw.pos_com_world) +
                   self.kd_com * (com_dot_des - pw.vel_com_world))

        # 2. CoMDynamics: K, u  (pw에서 발 위치 직접 읽음)
        self.com_dyn.update(pw, ddc_des, dL=dL_des)

        # 3. ForceOptimizer
        F_hat = self.force_opt.solve(self.com_dyn.K, self.com_dyn.u)

        # 4. WholeBodyTorqueGenerator  (pw에서 M, J_c 등 직접 읽음)
        tau = self.torque_gen.compute(pw, self.contact_keys[0], self.contact_keys[1], F_hat)

        return tau

    def get_desired_forces(self) -> dict:
        """
        Returns
        -------
        {contact_keys[0]: f(3,), contact_keys[1]: f(3,)}  목표 접촉력 [N], world frame
        키는 Pinocchio_Wrapper.fid 및 register_foot_geoms() 와 동일.
        """
        F = self.force_opt.opt_F  # (12,)
        return {
            self.contact_keys[0]: F[0:3].copy(),
            self.contact_keys[1]: F[6:9].copy(),
        }
