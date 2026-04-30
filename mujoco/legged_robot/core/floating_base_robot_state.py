import numpy as np
import pinocchio as pin


class FloatingBaseRobotState:
    '''
    nq = 36 (7+29), nv = 35, nu = 29
    Pinocchio 컨벤션: q = pos(3) + quat xyzw(4) + joints(29)
                     dq = lin_body(3) + ang_body(3) + joints(29)
    '''
    # 불변 데이터
    SLICES_Q = {
        "humanoid_base_pos": slice(0, 3),
        "humanoid_base_quad": slice(3, 7),
        "left_leg_angle": slice(7, 13),
        "right_leg_angle": slice(13, 19),
        "waist_angle": slice(19, 22),
        "left_arm_angle": slice(22, 29),
        "right_arm_angle": slice(29, 36)
    }

    SLICES_DQ = {
        "humanoid_base_lin_vel": slice(0, 3),
        "humanoid_base_ang_vel": slice(3, 6),
        "left_leg_vel": slice(6, 12),
        "right_leg_vel": slice(12, 18),
        "waist_vel": slice(18, 21),
        "left_arm_vel": slice(21, 28),
        "right_arm_vel": slice(28, 35)
        }
    
    def __init__(self):

        self.humanoid_base_pos = np.array([0.,  0.,  0.755]) # pos  3
        self.humanoid_base_quad = np.array([0., 0., 0., 1.]) # quat 4
        self.left_leg_angle = np.array([-0.312, 0., 0., 0.669, -0.363, 0.])  # 6DOF
        self.right_leg_angle = np.array([-0.312, 0., 0., 0.669, -0.363, 0.]) # 6DOF
        self.waist_angle = np.array([0.073, 0., 0.])  # 3DOF
        self.left_arm_angle = np.array([0., 0., 0., 0., 0., 0., 0.])  # 7DOF
        self.right_arm_angle = np.array([0., 0., 0., 0., 0., 0., 0.]) # 7DOF


        self.humanoid_base_lin_vel = np.zeros(3) # lin vel 3
        self.humanoid_base_ang_vel = np.zeros(3) # ang vel 3
        self.left_leg_vel = np.zeros(6)          # 6
        self.right_leg_vel = np.zeros(6)         # 6
        self.waist_vel = np.zeros(3)             # 3
        self.left_arm_vel = np.zeros(7)          # 7
        self.right_arm_vel = np.zeros(7)         # 7
        

    # G1, qpos = 36
    def update_floating_base_q(self, q):
        assert q.shape == (36,), f"q 차원 불일치 {q.shape}"

        for attr_name, slc in self.SLICES_Q.items():
            setattr(self, attr_name, q[slc].copy())
    
    # G1, qvel = 35
    def update_floating_base_dq(self, dq):
        assert dq.shape == (35,), f"dq 차원 불일치 {dq.shape}"

        for attr_name, slc in self.SLICES_DQ.items():
            setattr(self, attr_name, dq[slc].copy())

    def get_floating_base_q(self):
        return np.concatenate([getattr(self, attr) for attr in self.SLICES_Q.keys()])

    def get_floating_base_dq(self):
        return np.concatenate([getattr(self, attr) for attr in self.SLICES_DQ.keys()])