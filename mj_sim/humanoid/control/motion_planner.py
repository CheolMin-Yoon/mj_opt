import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core import Pinocchio_Wrapper
    from control import GaitScheduler, TrajOptimizer

'''
1. 어디를 밟을 것인가? 
compute_swing_traj_and_touchdown_humanoid

2. 웨이포인트는 어떻게? 
compute_bezier_trajectory
'''


class MotionPlanner:
    def __init__(self, wrapper: 'Pinocchio_Wrapper', traj_optimizer: 'TrajOptimizer', gait_scheduler: 'GaitScheduler'):
        
        self.wrapper = wrapper
        self.traj_optimizer = traj_optimizer
        self.gait_scheduler = gait_scheduler
        
        
    def compute_bezier_trajectory(self, phase, p_start, p_end, duration, step_height):
        """
        3D swing foot trajectory via 5th-order Bezier (Z) + 4th-order Bezier (XY).

        phase    : 0.0 ~ 1.0
        p_start  : (3,) swing start position [x, y, z]
        p_end    : (3,) swing end   position [x, y, z]
        duration : total swing time T [sec]
        step_height : peak height above ground [m]

        Returns
        -------
        pos : (3,) position  [x, y, z]
        vel : (3,) velocity  [m/s]
        acc : (3,) acceleration [m/s²]
        """
        t = phase
        omt = 1 - t

        # ── XY: 4차 Bezier, 제어점 [s, s, e, e, e] → 시작/끝 속도 0 ──
        # P_xy shape: (2, 5)
        xy_s = p_start[:2]
        xy_e = p_end[:2]
        Pxy = np.stack([xy_s, xy_s, (xy_s + xy_e) / 2, xy_e, xy_e])  # (5, 2)

        xy     = (omt**4 * Pxy[0] + 4*t*omt**3 * Pxy[1] + 6*t**2*omt**2 * Pxy[2] +
                  4*t**3*omt * Pxy[3] + t**4 * Pxy[4])

        dxy_dp = 4 * (omt**3 * (Pxy[1]-Pxy[0]) + 3*t*omt**2 * (Pxy[2]-Pxy[1]) +
                      3*t**2*omt * (Pxy[3]-Pxy[2]) + t**3 * (Pxy[4]-Pxy[3]))

        dxy2_dp2 = 12 * (omt**2 * (Pxy[2]-2*Pxy[1]+Pxy[0]) +
                         2*t*omt * (Pxy[3]-2*Pxy[2]+Pxy[1]) +
                         t**2 * (Pxy[4]-2*Pxy[3]+Pxy[2]))

        # ── Z: 5차 Bezier (기존 로직 유지) ──
        # h 두 제어점 비율로 궤적 가파름 튜닝 가능
        h = step_height
        Pz = np.array([p_start[2], p_start[2], h*1.2, h*0.5, p_end[2], p_end[2]])

        z      = (omt**5 * Pz[0] + 5*t*omt**4 * Pz[1] + 10*t**2*omt**3 * Pz[2] +
                  10*t**3*omt**2 * Pz[3] + 5*t**4*omt * Pz[4] + t**5 * Pz[5])

        dz_dp  = 5 * (omt**4 * (Pz[1]-Pz[0]) + 4*t*omt**3 * (Pz[2]-Pz[1]) +
                      6*t**2*omt**2 * (Pz[3]-Pz[2]) + 4*t**3*omt * (Pz[4]-Pz[3]) +
                      t**4 * (Pz[5]-Pz[4]))

        dz2_dp2 = 20 * (omt**3 * (Pz[2]-2*Pz[1]+Pz[0]) + 3*t*omt**2 * (Pz[3]-2*Pz[2]+Pz[1]) +
                        3*t**2*omt * (Pz[4]-2*Pz[3]+Pz[2]) + t**3 * (Pz[5]-2*Pz[4]+Pz[3]))

        # phase → 물리 시간 변환
        vel_xy = dxy_dp  / duration
        acc_xy = dxy2_dp2 / duration**2
        vel_z  = dz_dp   / duration
        acc_z  = dz2_dp2 / duration**2

        pos = np.array([xy[0],    xy[1],    z])
        vel = np.array([vel_xy[0], vel_xy[1], vel_z])
        acc = np.array([acc_xy[0], acc_xy[1], acc_z])

        return pos, vel, acc
    
    '''
    com의 위치와 속도 그리고 com_des 위치와 속도에 따른 다음 발걸음 위치를 계산하는 함수
    
    입력인자: 
            1. wrapper 즉 humanoid model로부터 base_pos, com_pos, vel를 받는다.
            2. traj_optimizer로부터 des_x_vel, des_y_vel, des_x_pos, des_y_pos를 받는다. 
            3. gait_scheduler로부터 leg를 받는다. 
            
    출력인자:
            1. 다음 스텝에 안넘어지기 위해서 딛어야할 pos_touchdown_world (3, ) 반환, 이때 z는 거의 0
    '''
    def compute_raibert_heuristic(self, leg: str):
        
        # wrapper로부터의 로봇상태
        curr_base_pos = self.wrapper.base_pos     # (3, )
        curr_com_pos = self.wrapper.com_pos_world # (3, )
        curr_com_vel = self.wrapper.com_vel_world # (3, )
        R_z = self.wrapper.R_z                    # (3x3)

        # 제어 목표치
        des_x_pos, des_y_pos = self.traj_optimizer.compute_desired_command()
        des_x_vel, des_y_vel = self.traj_optimizer.compute_desired_command()
        yaw_rate = self.traj_optimizer.compute_desired_command()
        
        # root 위치 계산 (골반)
        hip_offset = self.wrapper.get_hip_offset(leg) # (3, )
        hip_pos_world = curr_base_pos + R_z @ hip_offset   # (3, )
        
        # Capture Point 계산
        h_com = curr_com_pos[2]
        omega = np.sqrt(9.81 / h_com) 
        time_constant = 1.0 / omega   

        k_v_x = time_constant * 1.1  
        k_v_y = time_constant * 1.0  
        k_p_x = 0.05
        k_p_y = 0.05
        
        # 제자리 유지 -> 현재 hip의 x, y 좌표
        pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], 0.01]
        
        # com 위치 보정
        pos_correction_term = [
            k_p_x * (curr_com_pos[0] - des_x_pos), 
            k_p_y * (curr_com_pos[1] - des_y_pos), 
            0
        ]
        # com 속도 보정
        vel_correction_term = [
            k_v_x * (curr_com_vel[0] - des_x_vel), 
            k_v_y * (curr_com_vel[1] - des_y_vel), 
            0
        ]
        
        # yaw각 보정
        dtheta = yaw_rate * time_constant
        r_xy = np.array([pos_norminal_term[0], pos_norminal_term[1]]) - curr_base_pos[0:2]
        rotation_correction_term = np.array([
            -dtheta * r_xy[1],
            dtheta * r_xy[0],
            0.0
        ])
        
        # 최종 위치 산출
        pos_touchdown_world = (
            np.array(pos_norminal_term)
            + np.array(pos_correction_term)
            + np.array(vel_correction_term)
            + np.array(rotation_correction_term)
        )
        return pos_touchdown_world # (3, )
