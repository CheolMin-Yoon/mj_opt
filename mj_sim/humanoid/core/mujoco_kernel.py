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
    
    def register_foot_geoms(self, foot_body_map: dict):
        """
        발 body 이름 기반으로 접촉 가능한 geom id 집합을 사전 등록.
        geom에 name이 없는 xml에서도 동작한다.

        foot_body_map : {"R_foot": "right_ankle_roll_link", "L_foot": "left_ankle_roll_link"}
        호출 후 get_foot_contact_state() 사용 가능.
        """
        self._geom_id_to_key = {}
        for key, body_name in foot_body_map.items():
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for gid in range(self.model.ngeom):
                if (self.model.geom_bodyid[gid] == bid and
                        self.model.geom_contype[gid] > 0):
                    self._geom_id_to_key[gid] = key
        self._foot_keys = list(foot_body_map.keys())
        self._force_buf = np.zeros(6)

    def get_foot_contact_state(self) -> tuple:
        """
        전제: register_foot_geoms() 호출 완료.

        Returns
        -------
        forces : dict  {"R_foot": f_world(3,), "L_foot": f_world(3,)}  합력 [N]
        points : list  [np.ndarray(3,), ...]  접촉점 world 위치 (ConvexHull용)
        """
        forces = {k: np.zeros(3) for k in self._foot_keys}
        points = []

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            key = self._geom_id_to_key.get(c.geom1) or self._geom_id_to_key.get(c.geom2)
            if key is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, self._force_buf)
            forces[key] += c.frame.reshape(3, 3).T @ self._force_buf[:3]
            points.append(c.pos.copy())

        return forces, points


    # ===== sync to wrapper =====
    def push_to_wrapper(self, wrapper):
        q_mj, dq_mj = self.read_state()
        q_pin, dq_pin = self.mj_to_pin(q_mj, dq_mj)
        wrapper.update_model(q_pin, dq_pin)

    # ===== command =====
    def apply_torque_pin(self, tau_pin):
        tau_act = tau_pin[:]                # (29,)
        self.data.ctrl[self._tau_perm] = tau_act

    def apply_position_pin(self, q_des_pin):
        q_des_act = q_des_pin[7:]            # (29,)
        self.data.ctrl[self._tau_perm] = q_des_act

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)