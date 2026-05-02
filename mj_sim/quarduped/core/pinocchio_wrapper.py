'''
물리엔진인 Mujoco의 데이터를 Mujoco Kernel을 통해 받아 기구학/동역학 계산기인 Pinocchio를 이용해 계산된 제어입력을 전달하는 클래스
이 클래스의 입출력은 다음과 같다.

입력:
   1. EE 또는 manipulation 대상의 frame names을 받는다.
   2. Mujoco로부터 q, dq를 받는다. 

출력:
   1. world aligned jacobian, body jacobian, Jdot @ dq
   2. dynamcis M, nle, g, CoM, Centroidal Momentum ... (coriolis matrix는 pin API를 별도 호출 필요)
   3. world aligned된 ee의 state (SE3, Twist)
   4. world_R_base (SO3)
   5. CoM 기준 Moment Arm
   6. manipulation 대상의 base_T_root (SE3)

pinocchio로부터 정보를 가져오는 것은 view이므로 numpy 배열로 copy할 필요가 없다.
pinocchio C++로부터 데이터를 가져오는 것은 data.M 이런 식으로 복사하면 된다.

wrapper._q	pinocchio API 호출 인자 (computeAllTerms 등)
wrapper.q 외부 read-only 노출
'''


import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from .floating_base_robot_state import FloatingBaseRobotState

# Go2 frame — URDF link
GO2_FRAMES = {
    "FL": "FL_foot",
    "FR": "FR_foot",
    "RL": "RL_foot",
    "RR": "RR_foot",
    "FL_hip": "FL_hip",
    "FR_hip": "FR_hip",
    "RL_hip": "RL_hip",
    "RR_hip": "RR_hip",
    "base":   "base",
}


class Pinocchio_Wrapper:
    def __init__(self, urdf_path: str, package_dirs):
        robot = RobotWrapper.BuildFromURDF(
            str(urdf_path),
            package_dirs=[str(package_dirs)],
            root_joint=pin.JointModelFreeFlyer(),
        )

        self.model  = robot.model
        self.vmodel = robot.visual_model
        self.cmodel = robot.collision_model
        self.data   = self.model.createData()
        self.nv    = self.model.nv
        self.na   = self.model.nv - 6
        self.mass  = pin.computeTotalMass(self.model)
        self.current_state = FloatingBaseRobotState()

        # frame id 캐시
        self.fid = {k: self.model.getFrameId(v) for k, v in GO2_FRAMES.items()}

        # 초기 자세
        q_neutral = pin.neutral(self.model)
        pin.framesForwardKinematics(self.model, self.data, q_neutral)

        # 4발 hip placement (base_T_root) 
        oMb_neutral = self.data.oMf[self.fid["base"]]
        self.FL_hip_placement = oMb_neutral.actInv(self.data.oMf[self.fid["FL_hip"]]).copy()
        self.FR_hip_placement = oMb_neutral.actInv(self.data.oMf[self.fid["FR_hip"]]).copy()
        self.RL_hip_placement = oMb_neutral.actInv(self.data.oMf[self.fid["RL_hip"]]).copy()
        self.RR_hip_placement = oMb_neutral.actInv(self.data.oMf[self.fid["RR_hip"]]).copy()

        # 초기 상태
        self.q_init  = q_neutral.copy()
        self.dq_init = np.zeros(self.nv)

        # 마지막에 호출
        self.update_model(self.q_init, self.dq_init)

    def update_model(self, q, dq):
        self._q  = q
        self._dq = dq
        self.current_state.q = q
        self.current_state.dq = dq
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.ccrba(self.model, self.data, q, dq)

        # SE3
        self.oMb    = self.data.oMf[self.fid["base"]]
        self.oM_FL  = self.data.oMf[self.fid["FL"]]
        self.oM_FR  = self.data.oMf[self.fid["FR"]]
        self.oM_RL  = self.data.oMf[self.fid["RL"]]
        self.oM_RR  = self.data.oMf[self.fid["RR"]]

        # SO3
        self.R_body_to_world = self.oMb.rotation
        self.R_world_to_body = self.R_body_to_world.T
    
    '''
    pinocchio API에서 인자없이 가져올 함수들은 @property 활용
    '''
    @property
    def q(self):  return self._q
    @property
    def dq(self): return self._dq
    @property
    def M(self): return self.data.M
    @property
    def M_inv(self): return pin.computeMinverse(self.model, self.data, self._q)
    @property
    def C(self): return pin.computeCoriolisMatrix(self.model, self.data, self._q, self._dq)
    @property
    def nle(self): return self.data.nle
    @property
    def g(self): return self.data.g
    @property
    def pos_com_world(self): return self.data.com[0]
    @property
    def vel_com_world(self): return self.data.vcom[0]
    @property
    def hg(self): return self.data.hg
    @property
    def Ag(self): return self.data.Ag
    @property
    def angular_momentum(self): return self.data.hg[3:6]
    @property
    def R_z(self):
        """Base의 Yaw 회전만 추출한 SO3 행렬"""
        yaw = np.arctan2(self.R_body_to_world[1, 0], self.R_body_to_world[0, 0])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        return np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])

    # ── 자코비안 ─────────────────────────────────────────
    def _J(self, fid, ref):
        return pin.getFrameJacobian(self.model, self.data, fid, ref)

    def J_world(self, key: str):
        """world-aligned Jacobian. (lin(3,nv), ang(3,nv)) 반환."""
        J = self._J(self.fid[key], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J[:3], J[3:]

    def J_body(self, key: str):
        """body Jacobian. (lin(3,nv), ang(3,nv)) 반환."""
        J = self._J(self.fid[key], pin.ReferenceFrame.LOCAL)
        return J[:3], J[3:]

    def J_com(self):
        return pin.jacobianCenterOfMass(self.model, self.data, self._q)

    def Jdot_dq_world(self, key: str):
        Jd = pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.fid[key],
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return Jd @ self._dq

    def get_foot_placements_in_world(self):
        """4발 SE3 (FL, FR, RL, RR) 반환 (copy)."""
        return (self.oM_FL, self.oM_FR,
                self.oM_RL, self.oM_RR)

    def get_moment_arm_in_world(self, key: str):
        """key 프레임의 CoM 기준 모멘트 암 (world)."""
        p = self.data.oMf[self.fid[key]].translation
        return p - self.pos_com_world

    def ee_state_world(self, key: str):
        """발(또는 임의 frame)의 SE3 + 선/각속도 (world-aligned)."""
        fid = self.fid[key]
        oMf = self.data.oMf[fid].copy()
        twist = pin.getFrameVelocity(
            self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return oMf, twist.linear, twist.angular

    def world_to_base_frame(self, pos_world):
        """world 좌표계 pos → base local frame pos."""
        return self.R_world_to_body @ (pos_world - self.oMb.translation)

    def trajectory_world_to_base(self, traj_world):
        """(N, 3) world 궤적 → base frame 궤적."""
        p_base = self.oMb.translation
        R_wb   = self.R_world_to_body
        return (traj_world - p_base) @ R_wb.T
