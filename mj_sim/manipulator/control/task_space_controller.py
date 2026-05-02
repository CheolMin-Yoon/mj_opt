'''
'''


import numpy as np
import pinocchio as pin
from core import pinocchio_wrapper


class TaskSpaceController:
    def __init__(self, pinocchio_wrapper):
        self.wrapper = pinocchio_wrapper # pin wrapper 활용
        
        # 제어 파라미터
        self.I = np.eye(6)
        self.damping = 1e-3   
        wn_pos = 100.0   # 위치 자연진동수 (rad/s)
        wn_ori = 100.0   # 자세 자연진동수
        self.Kp_pos = wn_pos**2 * np.eye(3)   
        self.Kp_ori = wn_ori**2 * np.eye(3)   
        self.Kd_lin = 2 * wn_pos * np.eye(3) 
        self.Kd_ang = 2 * wn_ori * np.eye(3)  
        
        # 이 class의 controller가 필요한 변수 버퍼
        # target 관련은 trajectory 관련 class에서 설정
        self.pose_error = np.zeros(6)
        self.twist_error = np.zeros(6)
        self.desired_force = np.zeros(6)
        
    def compute_error(self, current_pos, current_R, current_twist, target_pos, target_R, target_twist):
        # pose error
        pos_error = target_pos - current_pos   # 병진은 그대로 계산
        R_error = target_R @ current_R.T       # 회전 오차 R*R.T
        ori_error = pin.log3(R_error)          # 로그변환 3x3 -> 3x1
        self.pose_error[:3] = pos_error
        self.pose_error[3:] = ori_error
        
        # twist error 
        self.twist_error[:3] = target_twist[:3] - current_twist[:3]  # 트위스트는 Lie 대수이므로 단순 계산 가능
        self.twist_error[3:] = target_twist[3:] - current_twist[3:]
        
        return self.pose_error, self.twist_error

               
    def compute_impedance_torque(self, pose_error, twist_error, J, M_inv, nle):
        Lambda_inv = J @ M_inv @ J.T
        Lambda = np.linalg.inv(Lambda_inv)
        F_pos = self.Kp_pos @ pose_error[:3] + self.Kd_lin @ twist_error[:3]
        F_ori = self.Kp_ori @ pose_error[3:] + self.Kd_ang @ twist_error[3:]
        self.desired_force[:3] = F_pos
        self.desired_force[3:] = F_ori
        tau = J.T @ (Lambda @ self.desired_force) + nle
        return tau
               

    def compute_DLS(self, target_twist, twist_error, J):
        V_pos_desired = target_twist[:3] + self.Kp_pos @ twist_error[:3] # 3x1
        V_ori_desired = target_twist[3:] + self.Kp_ori @ twist_error[3:] # 3x1
        V_desired = np.concatenate([V_pos_desired, V_ori_desired])
        JJt = J @ J.T + self.damping * self.I
        desired_dq = J.T @ np.linalg.solve(JJt, V_desired)  # 6x1
        return desired_dq