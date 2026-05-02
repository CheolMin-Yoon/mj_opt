import numpy as np
import mujoco
import pinocchio as pin

class Mujoco_Kernel:
    def __init__(self, xml_path, joint_names_pin_order=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self._force_buf = np.zeros(6)
        self._foot_keys = []
        self._geom_id_to_key = {}
        self.contact_forces = {}
        self._tau = np.zeros(self.model.nu)
        nq = self.model.nq
        nv = self.model.nv
        self._q_buf  = np.zeros(nq)
        self._dq_buf = np.zeros(nv)
        
        if not joint_names_pin_order or any(n is Ellipsis for n in joint_names_pin_order):
            raise ValueError("joint_names_pin_order를 명시적으로 전달해야 합니다.")    
        
        joint_id_to_act_id = {
            self.model.actuator_trnid[i, 0]: i for i in range(self.model.nu)
        }
        try:
            self._tau_perm = np.array([
                joint_id_to_act_id[self.model.joint(name).id]
                for name in joint_names_pin_order
            ], dtype=int)
        except KeyError as e:
            raise ValueError(f"Pinocchio 관절 {e} 가 MuJoCo Actuator에 연결되어 있지 않습니다!")

        # MJCF keyframe[0] 자세 (없으면 None)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.q_keyframe = self.data.qpos.copy()
            mujoco.mj_resetData(self.model, self.data)
        else:
            self.q_keyframe = None

    def reset_to_keyframe(self, key_id: int = 0):
        """MJCF keyframe 자세로 시뮬 초기화."""
        if self.model.nkey > key_id:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)


    # 상태 읽기
    @property
    def q_mj(self):
        return self.data.qpos
    @property
    def dq_mj(self):
        return self.data.qvel

    # quat 변환 후 state에 setter 경유 write
    def update_robot_state(self, state):
        q_mj  = self.q_mj
        dq_mj = self.dq_mj
        self._q_buf[0:3] = q_mj[0:3]
        self._q_buf[3]   = q_mj[4]
        self._q_buf[4]   = q_mj[5]
        self._q_buf[5]   = q_mj[6]
        self._q_buf[6]   = q_mj[3]
        self._q_buf[7:]  = q_mj[7:]
        quat = pin.Quaternion(self._q_buf[3:7])
        R_T = quat.toRotationMatrix().T
        self._dq_buf[0:3] = R_T @ dq_mj[0:3]
        self._dq_buf[3:]  = dq_mj[3:]
        state.q  = self._q_buf
        state.dq = self._dq_buf

    # 상태 쓰기 — pinocchio 순서(na,)의 tau를 MuJoCo ctrl에 in-place 반영
    @property
    def ctrl_tau(self):
        return self._tau

    @ctrl_tau.setter
    def ctrl_tau(self, tau_new):
        np.copyto(self._tau, tau_new)
        self.data.ctrl[self._tau_perm] = self._tau

    def step(self):
        mujoco.mj_step(self.model, self.data)
    
    # 접촉 geom names 등록
    def register_foot_geoms(self, foot_map: dict, by: str = "auto"):
        assert by in ("body", "geom", "auto"), f"by must be body/geom/auto, got {by}"

        for key, name in foot_map.items():
            registered = False
            if by in ("geom", "auto"):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    self._geom_id_to_key[gid] = key
                    registered = True

            if not registered and by in ("body", "auto"):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid < 0:
                    raise ValueError(f"register_foot_geoms: '{name}' not found")
                for gid in range(self.model.ngeom):
                    if (self.model.geom_bodyid[gid] == bid and self.model.geom_contype[gid] > 0):
                        self._geom_id_to_key[gid] = key
                        registered = True

            if not registered:
                raise ValueError(f"register_foot_geoms: '{name}' matched no geom")

        self._foot_keys = list(foot_map.keys())
        self.contact_forces = {k: np.zeros(3) for k in self._foot_keys}
    
    def get_foot_contact_state(self) -> tuple:
        """
        Returns
        -------
        forces : dict  {key: f_world(3,)}
        points : list  [np.ndarray(3,), ...]  실제 MuJoCo 접촉점 world 위치
        """
        for k in self._foot_keys:
            self.contact_forces[k].fill(0.0)
        points = []

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            key = self._geom_id_to_key.get(c.geom1) or self._geom_id_to_key.get(c.geom2)
            if key is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, self._force_buf)
            self.contact_forces[key] += c.frame.reshape(3, 3).T @ self._force_buf[:3]
            points.append(c.pos.copy())

        return self.contact_forces, points