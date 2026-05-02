'''
Pinocchio Wrapper의 데이터 인터페이스를 관리하는 상태 컨테이너 클래스
pinocchio (x, y, z, w)와 mujoco(w, x, y, z)의 quat 배열은 다른데 이는 Mujoco Kernel에서 변환 후 보낸다.
q  = pos(3) + quat xyzw(4) + joints(12) = 19
dq = lin_body(3) + ang_body(3) + joints(12) = 18

state._q view 유지
state.q copyto setter로 보호

Numpy에서 배열의 슬라이스를 변수에 할당하면 이는 원본 데이터의 특정 영역을 가리키는 pointer (view)를 생성
# np.copyto(A, B): A라는 메모리 주소는 그대로, 내부 값만 B로 갱신
'''


import numpy as np


class FloatingBaseRobotState:
    # class 변수, class 내 모든 객체가 공유
    NQ = 19
    NV = 18

    def __init__(self):
        self._q = np.zeros(self.NQ)
        self._dq = np.zeros(self.NV)

        self.base_pos    = self._q[0:3]
        self.base_quat   = self._q[3:7]
        self.FL_leg      = self._q[7:10]
        self.FR_leg      = self._q[10:13]
        self.RL_leg      = self._q[13:16]
        self.RR_leg      = self._q[16:19]

        self.base_lin_vel = self._dq[0:3]
        self.base_ang_vel = self._dq[3:6]
        self.FL_leg_vel   = self._dq[6:9]
        self.FR_leg_vel   = self._dq[9:12]
        self.RL_leg_vel   = self._dq[12:15]
        self.RR_leg_vel   = self._dq[15:18]
  
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