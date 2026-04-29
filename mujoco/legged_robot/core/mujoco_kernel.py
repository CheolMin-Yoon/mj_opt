import numpy as np
import mujoco
import pinocchio as pin


class Mujoco_Kernel:
    """
    MuJoCo 시뮬과 Pinocchio_Wrapper 브릿지 역할
    - state: read qpos/qvel → Pinocchio 컨벤션으로 변환 
    - command: Pinocchio τ(nv=35) → data.ctrl(nu=29) 이때 xml의 엑추에이터 타입 확인 필요!!
    """
    def __init__(self, xml_path, joint_names_pin_order=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        if not joint_names_pin_order or any(n is Ellipsis for n in joint_names_pin_order):
            raise ValueError(
                "joint_names_pin_order를 명시적으로 전달해야 합니다. "
                "(예: [pin_model.names[i] for i in range(2, pin_model.njoints)])"
            )    
        joint_id_to_act_id = {
            self.model.actuator_trnid[i, 0]: i for i in range(self.model.nu)
        }
        try:
            self._tau_perm = np.array([
                joint_id_to_act_id[self.model.joint(name).id]
                for name in joint_names_pin_order
            ], dtype=int)
        except KeyError as e:
            raise ValueError(f"Pinocchio 관절 {e} 가 MuJoCo Actuator에 연결되어 있지 않습니다! (XML 확인 필요)")

    # ===== read =====
    def read_state(self):
        return np.asarray(self.data.qpos).copy(), np.asarray(self.data.qvel).copy()

    @staticmethod
    def _quat_wxyz_to_xyzw(q4):
        return np.array([q4[1], q4[2], q4[3], q4[0]])

    @staticmethod
    def _quat_xyzw_to_wxyz(q4):
        return np.array([q4[3], q4[0], q4[1], q4[2]])

    @staticmethod
    def mj_to_pin(q_mj, dq_mj):
        # q: pos(3) + quat(4) + joints(29)
        q = q_mj.copy()
        q[3:7] = Mujoco_Kernel._quat_wxyz_to_xyzw(q_mj[3:7])
        quat = pin.Quaternion(np.array(q[3:7]))
        quat.normalize()  # 명시적 정규화
        R = quat.toRotationMatrix()
        dq = dq_mj.copy()
        # MuJoCo FreeFlyer: qvel[0:3] = world frame lin vel, qvel[3:6] = body frame ang vel
        # Pinocchio FreeFlyer: v[0:3] = body frame lin vel, v[3:6] = body frame ang vel
        dq[0:3] = R.T @ dq_mj[0:3]   # World -> Body 선속도 변환
        return q, dq

    @staticmethod
    def pin_to_mj(q_pin, dq_pin):
        q = q_pin.copy()
        quat = pin.Quaternion(np.array(q_pin[3:7]))
        quat.normalize()
        R = quat.toRotationMatrix()
        q_normalized = quat.coeffs()    # xyzw 정규화된 값
        q[3:7] = Mujoco_Kernel._quat_xyzw_to_wxyz(q_normalized)
        dq = dq_pin.copy()
        dq[0:3] = R @ dq_pin[0:3] 
        return q, dq

    # ===== sync to wrapper =====
    def push_to_wrapper(self, wrapper):
        q_mj, dq_mj = self.read_state()
        q_pin, dq_pin = self.mj_to_pin(q_mj, dq_mj)
        wrapper.update_model(q_pin, dq_pin)

    # ===== command =====
    def apply_torque_pin(self, tau_pin):
        tau_act = tau_pin[6:]                # (29,)
        self.data.ctrl[self._tau_perm] = tau_act

    def apply_position_pin(self, q_des_pin):
        q_des_act = q_des_pin[7:]            # (29,)
        self.data.ctrl[self._tau_perm] = q_des_act

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)