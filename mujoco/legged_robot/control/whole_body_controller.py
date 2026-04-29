import numpy as np
import pinocchio as pin
import osqp
import proxsuite
import scipy.sparse as sp


# =============================================================================
# 1. WholeBodyTorqueGenerator  (from whole_body_torque.cpp)
#    G z = f  →  z* = (G^T G + W)^{-1} G^T f  →  tau = z*[nv:]
# =============================================================================

class WholeBodyTorqueGenerator:
    """
    Floating-base dynamics + contact constraint 연립으로 tau를 직접 계산.

    G = [M(q)   -S^T ]    f = [ -nle + J_c^T F_hat ]
        [J_c(q)   0  ]        [ -Jdot_qdot          ]

    z = [ddq(nv), tau(na)],  z* = (G^T G + W)^{-1} G^T f
    """

    W_DDQ = 1e-4
    W_TAU = 1e-3

    def __init__(self, nv: int, na: int, nc: int):
        self.nv, self.na, self.nc = nv, na, nc

        self.G = np.zeros((nv + nc, nv + na))
        self.f = np.zeros(nv + nc)

        self.W = np.zeros((nv + na, nv + na))
        self.W[:nv, :nv] = np.eye(nv) * self.W_DDQ
        self.W[nv:, nv:] = np.eye(na) * self.W_TAU

        # -S^T: floating base(6) = 0, actuated(na) = -I  (고정)
        self.G[6:nv, nv:] = -np.eye(na)

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

        # contact Jacobian (update_model에서 이미 computeJointJacobiansTimeVariation 완료)
        J_rf = pin.getFrameJacobian(model, data, rf_id, pin.LOCAL_WORLD_ALIGNED)
        J_lf = pin.getFrameJacobian(model, data, lf_id, pin.LOCAL_WORLD_ALIGNED)
        self.J_c[:6]  = J_rf
        self.J_c[6:]  = J_lf

        # Jdot*qdot (발이 땅에 고정 → desired classical acc = 0)
        self.Jdot_qdot[:6] = pin.getFrameClassicalAcceleration(
            model, data, rf_id, pin.LOCAL_WORLD_ALIGNED).vector
        self.Jdot_qdot[6:] = pin.getFrameClassicalAcceleration(
            model, data, lf_id, pin.LOCAL_WORLD_ALIGNED).vector

        # wrapper에서 동역학 항 꺼내기 (M, g, nle 순서)  nle = C@dq + g
        M, _, nle = pw.compute_dynamics_term()

        # G 블록 조립
        self.G[:self.nv, :self.nv] = M
        self.G[self.nv:, :self.nv] = self.J_c

        self.f[:self.nv] = -nle + self.J_c.T @ F_hat
        self.f[self.nv:] = -self.Jdot_qdot

        # z* = (G^T G + W)^{-1} G^T f
        z = np.linalg.solve(self.G.T @ self.G + self.W, self.G.T @ self.f)

        return z[self.nv:]   # tau


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

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        return np.array([[0, -v[2], v[1]],
                         [v[2],  0, -v[0]],
                         [-v[1], v[0], 0]])

    def update(self, pw, ddc_des: np.ndarray, dL: np.ndarray = None):
        """
        pw      : Pinocchio_Wrapper (update_model 완료 상태)
        ddc_des : (3,) desired CoM acceleration
        dL      : (3,) rate of angular momentum (default 0)
        """
        if dL is None:
            dL = np.zeros(3)

        com_pos = pw.pos_com_world
        rf_pos  = pw.oM_Rfoot.translation
        lf_pos  = pw.oM_Lfoot.translation

        r_R = rf_pos - com_pos
        r_L = lf_pos - com_pos

        D2 = np.zeros((3, 12))
        D2[:, 0:3]  = self._skew(r_R)
        D2[:, 3:6]  = np.eye(3)
        D2[:, 6:9]  = self._skew(r_L)
        D2[:, 9:12] = np.eye(3)

        self.K[:3] = self._D1
        self.K[3:]  = D2

        F_g = np.array([0.0, 0.0, self.m * self.g])
        self.u[:3] = self.m * ddc_des + F_g
        self.u[3:] = dL


# =============================================================================
# 3. ForceOptimizer  (from Force_Optimizer.cpp + friction_cone.cpp)
#    min ||KF - u||^2 + F^T W_reg F   s.t. friction cone per foot
# =============================================================================

class ForceOptimizer:
    """
    OSQP 기반 접촉력 최적화.
    F (12,) = [f_R(3), tau_R(3), f_L(3), tau_L(3)]

    마찰원뿔 (linearized, mu/sqrt(2) 근사):
        ±fx <= mu_eff * fz
        ±fy <= mu_eff * fz
        0 <= fz <= fz_max
    """

    def __init__(self, mu: float = 0.5, fz_max: float = 500.0):
        self.mu = mu
        self.fz_max = fz_max
        self.n_vars = 12
        self._solver = None
        self._initialized = False
        self.opt_F = np.zeros(self.n_vars)

    def _build_friction_block(self) -> tuple:
        mu_eff = self.mu / np.sqrt(2.0)
        INF = 1e9
        # force 3열에만 마찰원뿔, torque 3열은 unconstrained
        A3 = np.array([[ 1,  0, -mu_eff],
                       [-1,  0, -mu_eff],
                       [ 0,  1, -mu_eff],
                       [ 0, -1, -mu_eff],
                       [ 0,  0,  1     ]])
        l3 = np.array([-INF, -INF, -INF, -INF, 0.0])
        u3 = np.array([0.0,  0.0,  0.0,  0.0, self.fz_max])

        A6 = np.zeros((5, 6))
        A6[:, :3] = A3
        return A6, l3, u3

    def _build_constraints(self) -> tuple:
        A6, l3, u3 = self._build_friction_block()
        A = np.zeros((10, self.n_vars))
        A[:5,  0:6]  = A6
        A[5:, 6:12]  = A6
        return sp.csc_matrix(A), np.concatenate([l3, l3]), np.concatenate([u3, u3])

    def solve(self, K: np.ndarray, u_vec: np.ndarray,
              W_reg: np.ndarray = None) -> np.ndarray:
        """
        K     : (6, 12)
        u_vec : (6,)
        returns opt_F (12,)
        """
        if W_reg is None:
            W_reg = 1e-4 * np.eye(self.n_vars)

        P = sp.csc_matrix(2.0 * (K.T @ K + W_reg))
        q = -2.0 * (K.T @ u_vec)
        A_sp, l_con, u_con = self._build_constraints()

        if not self._initialized:
            self._solver = osqp.OSQP()
            self._solver.setup(P, q, A_sp, l_con, u_con,
                               warm_starting=True,
                               eps_abs=1e-4, eps_rel=1e-4,
                               max_iter=1000, verbose=False)
            self._initialized = True
        else:
            self._solver.update(Px=P.data, q=q,
                                Ax=A_sp.data, l=l_con, u=u_con)

        res = self._solver.solve()
        if res.info.status == 'solved':
            self.opt_F = res.x

        return self.opt_F


# =============================================================================
# 3-B. ForceOptimizerProx  (ProxQP 버전)
#      미리 크기 고정 + warm-start로 실시간 성능 확보
# =============================================================================

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
        returns opt_F (12,)
        """
        if W_reg is None:
            W_reg = 1e-4 * np.eye(self.N_VARS)

        H = 2.0 * (K.T @ K + W_reg)
        g = -2.0 * (K.T @ u_vec)

        if not self._initialized:
            # 첫 호출: 실제 H, g로 재init (warm-start 기준점 설정)
            self._qp.init(H, g, None, None, self._A, self._l, self._u)
            self._initialized = True
        else:
            # 이후: H, g만 교체, constraint는 불변이므로 생략 가능
            self._qp.update(H=H, g=g)

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

    KP_COM = 100.0
    KD_COM = 20.0

    def __init__(self, pw, nc: int = 12, mu: float = 0.5, solver: str = "proxqp"):
        """
        pw     : Pinocchio_Wrapper
        nc     : contact DOF (6 per foot × 2 = 12)
        solver : "proxqp" (기본, warm-start) | "osqp"
        """
        self.pw = pw
        self.torque_gen = WholeBodyTorqueGenerator(pw.nv, pw.na, nc)
        self.com_dyn    = CoMDynamics(pw.mass)
        if solver == "proxqp":
            self.force_opt = ForceOptimizerProx(mu=mu)
        else:
            self.force_opt = ForceOptimizer(mu=mu)

    def compute(self, com_des: np.ndarray, com_dot_des: np.ndarray) -> np.ndarray:
        """
        com_des     : (3,) desired CoM position  [world frame]
        com_dot_des : (3,) desired CoM velocity  [world frame]
        returns     : tau (na,)

        전제: pw.update_model(q, dq) 가 이미 호출된 상태.
        """
        pw = self.pw

        # 1. BalanceTask: PD → ddc_des
        ddc_des = (self.KP_COM * (com_des     - pw.pos_com_world) +
                   self.KD_COM * (com_dot_des - pw.vel_com_world))

        # 2. CoMDynamics: K, u  (pw에서 발 위치 직접 읽음)
        self.com_dyn.update(pw, ddc_des)

        # 3. ForceOptimizer
        F_hat = self.force_opt.solve(self.com_dyn.K, self.com_dyn.u)

        # 4. WholeBodyTorqueGenerator  (pw에서 M, J_c 등 직접 읽음)
        tau = self.torque_gen.compute(pw, "R_foot", "L_foot", F_hat)

        return tau
