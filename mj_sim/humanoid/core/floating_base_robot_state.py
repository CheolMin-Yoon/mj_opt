import numpy as np


class FloatingBaseRobotState:

    NQ = 36
    NV = 35

    def __init__(self):
        self._q = np.zeros(self.NQ)
        self._dq = np.zeros(self.NV)

        self.humanoid_base_pos = self._q[0:3]        # pos 3
        self.humanoid_base_quad = self._q[3:7]       # quat 4
        self.left_leg_angle = self._q[7:13]          # 6DOF
        self.right_leg_angle = self._q[13:19]        # 6DOF
        self.waist_angle = self._q[19:22]            # 3DOF
        self.left_arm_angle = self._q[22:29]         # 7DOF
        self.right_arm_angle = self._q[29:36]        # 7DOF

        self.humanoid_base_lin_vel = self._dq[0:3]   # lin 3
        self.humanoid_base_ang_vel = self._dq[3:6]   # ang 3
        self.left_leg_vel = self._dq[6:12]           # 6
        self.right_leg_vel = self._dq[12:18]         # 6
        self.waist_vel = self._dq[18:21]             # 3
        self.left_arm_vel = self._dq[21:28]          # 7
        self.right_arm_vel = self._dq[28:35]         # 7

        @property
        def q(self):
            return self._q
        @q.setter
        def q(self, q_update):
            np.copyto(self._q, q_update)
        @property
        def dq(self):
            return self._dq
        @dq.setter
        def dq(self, dq_update):
            np.copyto(self._dq, dq_update)