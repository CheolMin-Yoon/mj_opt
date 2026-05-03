import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from .floating_base_robot_state import FloatingBaseRobotState

# G1 frame — URDF link
G1_FRAMES = {
    "L_foot": "left_ankle_roll_link",
    "R_foot": "right_ankle_roll_link",
    "L_hand": "left_rubber_hand",
    "R_hand": "right_rubber_hand",
    "L_hip" : "left_hip_pitch_link",
    "R_hip" : "right_hip_pitch_link",
    "L_sh"  : "left_shoulder_pitch_link",
    "R_sh"  : "right_shoulder_pitch_link",
    "base"  : "pelvis",
}


class Pinocchio_Wrapper:

    def __init__(self, urdf_path: str, package_dirs):
        robot = RobotWrapper.BuildFromURDF(
            str(urdf_path),
            package_dirs = [str(package_dirs)],
            root_joint = pin.JointModelFreeFlyer() # 부유 모델 
        )
      
        self.model = robot.model
        self.vmodel = robot.visual_model
        self.cmodel = robot.collision_model
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.na = self.model.nv - 6   # actuated joints (floating base 6 제외)
        self.mass = pin.computeTotalMass(self.model)
        self.current_state = FloatingBaseRobotState()
        
        self.fid = {k: self.model.getFrameId(v) for k, v in G1_FRAMES.items()}

        # 초기 자세
        q_neutral = pin.neutral(self.model)
        pin.framesForwardKinematics(self.model, self.data, q_neutral)

        # SE3 
        oMb_neutral = self.data.oMf[self.fid["base"]]
        self.L_hip_placement = oMb_neutral.actInv(self.data.oMf[self.fid["L_hip"]]).copy()
        self.R_hip_placement = oMb_neutral.actInv(self.data.oMf[self.fid["R_hip"]]).copy()
        self.L_shoulder_placement = oMb_neutral.actInv(self.data.oMf[self.fid["L_sh"]]).copy()
        self.R_shoulder_placement = oMb_neutral.actInv(self.data.oMf[self.fid["R_sh"]]).copy()
        
        # 초기 상태
        self.q_init = q_neutral.copy()
        self.dq_init = np.zeros(self.nv)
    
        self.update_model(self.q_init, self.dq_init)

    
    # pin model update
    def update_model(self, q, dq):
        self._q  = q
        self._dq = dq
        self.current_state.q = q
        self.current_state.dq = dq
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.ccrba(self.model, self.data, q, dq)
        pin.centerOfMass(self.model, self.data, q, dq)

        # SE3
        self.oMb      = self.data.oMf[self.fid["base"]]
        self.oM_Lfoot = self.data.oMf[self.fid["L_foot"]]
        self.oM_Rfoot = self.data.oMf[self.fid["R_foot"]]
        self.oM_Lhand = self.data.oMf[self.fid["L_hand"]]
        self.oM_Rhand = self.data.oMf[self.fid["R_hand"]]

        # SO3
        self.R_body_to_world = self.oMb.rotation
        self.R_world_to_body = self.R_body_to_world.T


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
    def base_pos(self): return self.data.oMf[self.fid["base"]].translation # (3, )
    @property
    def com_pos_world(self): return self.data.com[0] # (3, )
    @property
    def com_vel_world(self): return self.data.vcom[0] # (3, )
    @property
    def hg(self): return self.data.hg
    @property
    def Ag(self): return self.data.Ag
    @property
    def angular_momentum(self): return self.data.hg[3:6]
    @property
    def R_z(self): 
        """Base의 Yaw 회전만 추출한 SO3 행렬""" # (3x3)
        yaw = np.arctan2(self.R_body_to_world[1, 0], self.R_body_to_world[0, 0])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        return np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
    

    def _J(self, fid, ref):
        return pin.getFrameJacobian(self.model, self.data, fid, ref)

    def J_world(self, key: str):
        J = self._J(self.fid[key], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J

    def J_body(self, key: str):
        J = self._J(self.fid[key], pin.ReferenceFrame.LOCAL)
        return J

    def J_com(self):
        return pin.jacobianCenterOfMass(self.model, self.data, self._q)

    def Jdot_dq_world(self, key: str):
        Jd = pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.fid[key],
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return Jd @ self._dq
    
    def get_moment_arm_in_world(self, key: str):
        fid = self.fid[key]
        p_ee = self.data.oMf[fid].translation
        moment_arm = p_ee - self.pos_com_world
        return moment_arm
    
    def get_ee_state_world(self, key: str):
        fid = self.fid[key]
        oMf = self.data.oMf[fid]
        twist = pin.getFrameVelocity(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return oMf, twist.linear, twist.angular
    
    def world_to_base_frame(self, pos_world):
        return self.R_world_to_body @ (pos_world - self.oMb.translation)

    def trajectory_world_to_base(self, traj_world):
        p_base = self.oMb.translation
        R_wb   = self.R_world_to_body
        return (traj_world - p_base) @ R_wb.T
    
    def get_hip_offset(self, leg):
        prefix = 'L' if 'left' in leg.lower() or 'l' == leg.lower() else 'R'
        placement = getattr(self, f"{prefix}_hip_placement")
        return placement.translation

    
    