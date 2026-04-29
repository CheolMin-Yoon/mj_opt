import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from .floating_base_robot_state import FloatingBaseRobotState


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
        
        # 딕셔너리에 의존성 존재
        names = {
            "L_foot": "left_ankle_roll_link",
            "R_foot": "right_ankle_roll_link",
            "L_hand": "left_rubber_hand",
            "R_hand": "right_rubber_hand",
            "L_hip":  "left_hip_pitch_link",
            "R_hip":  "right_hip_pitch_link",
            "L_sh":   "left_shoulder_pitch_link",
            "R_sh":   "right_shoulder_pitch_link",
            "base":   "pelvis",
        }
        self.fid = {k: self.model.getFrameId(v) for k, v in names.items()}

        # 초기 상태 가져오기 (URDF로부터)
        q_neutral = pin.neutral(self.model)
        pin.framesForwardKinematics(self.model, self.data, q_neutral)

        # 하드웨어 정보 저장
        oMb_neutral = self.data.oMf[self.fid["base"]]
        self.L_hip_placement = oMb_neutral.actInv(self.data.oMf[self.fid["L_hip"]]).copy()
        self.R_hip_placement = oMb_neutral.actInv(self.data.oMf[self.fid["R_hip"]]).copy()
        self.L_shoulder_placement = oMb_neutral.actInv(self.data.oMf[self.fid["L_sh"]]).copy()
        self.R_shoulder_placement = oMb_neutral.actInv(self.data.oMf[self.fid["R_sh"]]).copy()
        
        # 초기 상태 저장
        self.q_init = self.current_state.get_floating_base_q()
        self.dq_init = self.current_state.get_floating_base_dq()
        self._dq_cache = np.zeros(self.nv)
    
        # 마지막에 있어야함
        self.update_model(self.q_init, self.dq_init)

    
    # pinocchio 업데이트
    def update_model(self, q, dq):
        self.current_state.update_floating_base_q(q)      # 관절각 업데이트
        self.current_state.update_floating_base_dq(dq)    # 관절속도 업데이트
        self._dq_cache = dq.copy()
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.ccrba(self.model, self.data, q, dq)
        pin.centerOfMass(self.model, self.data, q, dq)
        pin.computeTotalMass(self.model)

        # world_T_base 업데이트
        self.oMb = self.data.oMf[self.fid["base"]]   
        
        # world_T_ee 업데이트
        self.oM_Lfoot = self.data.oMf[self.fid["L_foot"]]
        self.oM_Rfoot = self.data.oMf[self.fid["R_foot"]]
        self.oM_Lhand = self.data.oMf[self.fid["L_hand"]]
        self.oM_Rhand = self.data.oMf[self.fid["R_hand"]]

        # world 기준 전신 CoM 위치와 속도 업데이트
        self.pos_com_world = self.data.com[0].copy()
        self.vel_com_world = self.data.vcom[0].copy()

        # world 기준 Centroidal Momentum Matrix
        self.Ag = self.data.Ag.copy()    # centroidal momentum Matrix (6 x nv)
        self.hg = self.data.hg.copy()     # centroidal momentum (6 x 1)

        # world_R_body, body_R_world (SO3) 업데이트
        R_bw = self.oMb.rotation.copy()
        self.R_body_to_world = R_bw
        self.R_world_to_body = R_bw.T
    
    # 동역학 계산
    def compute_dynamics_term(self):
        M = self.data.M.copy()
        nle = self.data.nle.copy()
        g = self.data.g.copy()
        return M, g, nle
    
    def compute_coriolis_matrix(self):
        q = self.current_state.get_floating_base_q()
        dq = self._dq_cache
        pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        return self.data.C.copy()
    
    def compute_centroidal_term(self):
        return self.Ag, self.hg 
    
    def angular_momentum(self):
        return self.hg[3:6] 
    
    # 자코비안 계산 
    def _J(self, fid, ref):
        return pin.getFrameJacobian(self.model, self.data, fid, ref)
    
    # 월드 기준 정렬된 자코비안 (Modern Robotics의 space Jacobian이 아니다)
    def J_world(self, key: str):
        J = self._J(self.fid[key], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J[:3], J[3:] # (lin, ang)
    
    # 바디 기준 자코비안 (Modern Robotics의 J_body랑 동일하다)
    def J_body(self, key: str):
        J = self._J(self.fid[key], pin.ReferenceFrame.LOCAL)
        return J[:3], J[3:]
    
    # CoM Jacobian
    def J_com(self):
        return pin.jacobianCenterOfMass(self.model, self.data, self.current_state.get_floating_base_q())
    
    # dJ 
    def Jdot_dq_world(self, key: str):
        Jd = pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.fid[key],
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return Jd @ self._dq_cache

    # 월드 기준 엔드이펙터들의 Transformation Matrix (SE3)
    def get_ee_placement_in_world(self):
        return (self.oM_Lfoot.copy(), self.oM_Rfoot.copy(),
                self.oM_Lhand.copy(), self.oM_Rhand.copy())
    
    #특정 프레임의 모멘트 암 반환 
    def get_moment_arm_in_world(self, key: str):
        fid = self.fid[key]
        p_ee = self.data.oMf[fid].translation
        moment_arm = p_ee - self.pos_com_world
        return moment_arm
    
    # 월드 기준 엔드이펙터의 위치/회전(SE3) 및 선형/각속도 반환
    def ee_state_world(self, key: str):
        fid = self.fid[key]
        oMf = self.data.oMf[fid].copy()
        twist = pin.getFrameVelocity(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return oMf, twist.linear.copy(), twist.angular.copy()
    
    # 
    def world_to_base_frame(self, pos_world):
        """world 좌표계 pos → base local frame pos."""
        return self.R_world_to_body @ (pos_world - self.oMb.translation)
    
    # (N, 3) world 궤적 → base frame 궤적
    def trajectory_world_to_base(self, traj_world):
        p_base = self.oMb.translation
        R_wb   = self.R_world_to_body
        return (traj_world - p_base) @ R_wb.T

    
    