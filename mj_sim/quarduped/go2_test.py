# %%
import sys
sys.path.insert(0, '/home/frlab/mj_opt/mj_sim/quarduped')

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial import ConvexHull
from dataclasses import dataclass

from core import FloatingBaseRobotState, Pinocchio_Wrapper, Mujoco_Kernel
from utils import SimScheduler, Visualizer

# %%
URDF = '/home/frlab/mj_opt/xmls/robots/go2_description/urdf/go2_description.urdf'
PKG  = '/home/frlab/mj_opt/xmls/robots'
MJCF = '/home/frlab/mj_opt/xmls/systems/go2/scene.xml'

# %%
PHASE_OFFSET = np.array([0.5, 0.0, 0.0, 0.5])   # trot: FL+RR vs FR+RL
HEIGHT_SWING = 0.08
LEG_KEYS     = ["FL", "FR", "RL", "RR"]
LEG_IDX      = {k: i for i, k in enumerate(LEG_KEYS)}


class GaitScheduler:
    def __init__(self, gait_hz: float, duty: float):
        self.gait_period  = 1.0 / gait_hz
        self.gait_duty    = duty
        self.stance_time  = duty * self.gait_period
        self.swing_time   = (1.0 - duty) * self.gait_period

    def get_mask(self, t: float) -> np.ndarray:
        """현재 시각 t에서 4발 contact mask 반환. 1=stance, 0=swing."""
        phases = np.mod(PHASE_OFFSET + t / self.gait_period, 1.0)
        return (phases < self.gait_duty).astype(np.int32)

    def make_swing_traj(self, p0: np.ndarray, pf: np.ndarray, h_sw: float):
        """이륙점 p0 → 착지점 pf 의 5차 + bump z 궤적 클로저 반환."""
        p0 = np.array(p0, dtype=float)
        pf = np.array(pf, dtype=float)
        dp = pf - p0
        T  = self.swing_time

        def eval_at(dt: float):
            s    = np.clip(dt / T, 0.0, 1.0)
            poly  = 10*s**3 - 15*s**4 + 6*s**5
            dpoly = (30*s**2 - 60*s**3 + 30*s**4) / T
            d2poly= (60*s   - 180*s**2 + 120*s**3) / T**2
            p = p0 + dp * poly
            v = dp * dpoly
            a = dp * d2poly
            if h_sw > 0.0:
                b   =  64*s**3*(1-s)**3
                db  = (192*s**2*(1-s)**2*(1-2*s)) / T
                d2b = (192*(2*s*(1-s)**2*(1-2*s)
                           - 2*s**2*(1-s)*(1-2*s)
                           - 2*s**2*(1-s)**2)) / T**2
                p[2] += h_sw * b
                v[2] += h_sw * db
                a[2] += h_sw * d2b
            return p, v, a

        return eval_at


@dataclass
class LegOutput:
    tau:     np.ndarray   # (3,) joint torque
    pos_des: np.ndarray
    pos_now: np.ndarray
    vel_des: np.ndarray
    vel_now: np.ndarray


class LegController:
    KP_STANCE = np.diag([80.0, 80.0, 80.0])
    KD_STANCE = np.diag([10.0, 10.0, 10.0])

    # swing: task-space (발끝 world) PD gains
    KP_SWING  = np.diag([100.0, 100.0, 100.0])
    KD_SWING  = np.diag([10.0,  10.0,   10.0])

    def __init__(self, wrapper: Pinocchio_Wrapper, gait: GaitScheduler, kernel: Mujoco_Kernel):
        self.wrapper    = wrapper
        self.gait       = gait
        self._last_mask = np.ones(4, dtype=np.int32)   # 초기: 전부 stance
        self._traj      = {}    # leg -> eval_at 클로저
        self._t_takeoff = {}    # leg -> 이륙 시각

        # stance nominal: MJCF keyframe[0] joint 자세 (pinocchio neutral과 다름)
        if kernel.q_keyframe is not None:
            self._q_nom = kernel.q_keyframe[7:].copy()   # (12,)
        else:
            self._q_nom = np.zeros(12)

    def compute(self, t: float) -> np.ndarray:
        wrapper = self.wrapper
        gait    = self.gait
        mask    = gait.get_mask(t)
        tau_all = np.zeros(12)

        for leg in LEG_KEYS:
            idx = LEG_IDX[leg]
            col = idx * 3   # 0,3,6,9 for FL,FR,RL,RR

            J_lin, _ = wrapper.J_world(leg)
            J3 = J_lin[:, 6:][:, col : col + 3]   # (3, 3) 해당 다리 3DOF

            oMf, v_lin, _ = wrapper.ee_state_world(leg)
            pos_now = oMf.translation
            vel_now = v_lin

            if mask[idx] == 0:   # ── swing ──
                if self._last_mask[idx] == 1:
                    p0 = pos_now.copy()
                    pf = p0.copy()
                    self._traj[leg]      = gait.make_swing_traj(p0, pf, HEIGHT_SWING)
                    self._t_takeoff[leg] = t

                dt     = t - self._t_takeoff.get(leg, t)
                eval_at = self._traj.get(leg)
                if eval_at is None:
                    continue

                pos_des, vel_des, _ = eval_at(dt)
                f_task = self.KP_SWING @ (pos_des - pos_now) + self.KD_SWING @ (vel_des - vel_now)
                tau3   = J3.T @ f_task

            else:   # ── stance ──
                q_now  = wrapper.q[7 + col : 7 + col + 3]
                dq_now = wrapper.dq[6 + col : 6 + col + 3]
                q_des  = self._q_nom[col : col + 3]
                tau3   = self.KP_STANCE @ (q_des - q_now) - self.KD_STANCE @ dq_now

            tau_all[col : col + 3] = tau3
            self._last_mask[idx] = mask[idx]

        return tau_all


# %%
state    = FloatingBaseRobotState()
wrapper  = Pinocchio_Wrapper(URDF, PKG)
joint_names = [wrapper.model.names[i] for i in range(2, wrapper.model.njoints)]
kernel   = Mujoco_Kernel(MJCF, joint_names_pin_order=joint_names)
kernel.register_foot_geoms({"FL": "FL", "FR": "FR", "RL": "RL", "RR": "RR"})

gait     = GaitScheduler(gait_hz=1.0, duty=0.7)
leg_ctrl = LegController(wrapper, gait, kernel)

print("✅ 시스템 초기화 완료")


# %%
def on_control(t):
    kernel.update_robot_state(state)
    wrapper.update_model(state.q, state.dq)

    tau = leg_ctrl.compute(t)
    kernel.ctrl_tau = tau


import numpy as np
from scipy.spatial import ConvexHull

_dbg_count = 0

def on_render(t, visualizer):
    global _dbg_count

    # 1. 접촉 데이터 가져오기
    _, points = kernel.get_foot_contact_state()
    
    # ---------------------------------------------------------
    # [1] 3발 이상 접촉: 지지 다각형 (Support Polygon) - 초록색
    # ---------------------------------------------------------
    if len(points) >= 3:
        pts = np.array(points)
        try:
            hull = ConvexHull(pts[:, :2])
            z = pts[:, 2].mean()
            verts = pts[hull.vertices]
            num_verts = len(verts)
            for j in range(num_verts):
                p1 = np.array([verts[j][0], verts[j][1], z])
                p2 = np.array([verts[(j+1) % num_verts][0],
                               verts[(j+1) % num_verts][1], z])
                visualizer.draw_line(p1, p2, rgba=(0, 1, 0, 1), thickness=0.005)
        except Exception:
            pass

    # ---------------------------------------------------------
    # [2] 딱 2발만 접촉 (Trot 보행 중): 지지선 (Support Line) - 파란색
    # ---------------------------------------------------------
    elif len(points) == 2:
        p1 = np.array(points[0])
        p2 = np.array(points[1])
        
        # 선이 땅에 묻히지 않게 두 점의 평균 높이 사용
        z_height = (p1[2] + p2[2]) / 2.0
        p1[2] = z_height
        p2[2] = z_height
        
        # 파란색(0, 0, 1, 1) 실선으로 대각선 지지선을 그립니다.
        visualizer.draw_line(p1, p2, rgba=(0, 0, 1, 1), thickness=0.005)


# %%
kernel.reset_to_keyframe()
sched = SimScheduler(kernel.model, kernel.data, ctrl_hz=500, render_hz=60)
sched.run(on_control=on_control, on_render=on_render)
