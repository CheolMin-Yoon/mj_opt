import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicSpline


class MotionPlanner:


    def __init__(self, default_step_height=0.05):
        
        self.default_step_height = default_step_height
        

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


    def compute_se3_spline_interpolator(self, wp_pos, wp_rot, seg_duration, dt):
        '''
        주목적, waypoints를 받아서 seg_duration 동안 dt(low-level-control-rate)만큼의 보간
        입력은 공간에서의 pos, rot를 받아서 dt에 대한 보간 진행 -> torque? 
        '''
        T_wp = [pin.SE3(R, p) for R, p in zip(wp_rot, wp_pos)] # (3, ), (3, 3)을 가지고 (4, 4) 생성
        T_ref = T_wp[0]
        xi_wp = np.array([pin.log6(T_ref.inverse() * Ti).vector for Ti in T_wp]) # SE(3) 공간으로 펴는 작업

        t_wp = np.arange(len(T_wp)) * seg_duration # waypoints time 전체 궤적의 시간
        cs = CubicSpline(t_wp, xi_wp, bc_type='clamped') # slpine 곡선의 경계조건이 clamped로 도착점에서의 속도=0
        t_s = np.arange(0, t_wp[-1], dt) # 이건 아마 궤적생성할 때 쓰는 시간도메인 
        xi = cs(t_s)      # (N, 6) 
        xi_d = cs(t_s, 1) # (N, 6) ← body twist

        pos = np.zeros((len(t_s), 3))      # t_s만큼 쭉 편 (3, ) pos 데이터들
        rot = np.zeros((len(t_s), 3, 3))   # t_s만큼 쭉 편 (3, 3) R 데이터들
        for i, v in enumerate(xi):
            T_t = T_ref * pin.exp6(pin.Motion(v))
            pos[i] = T_t.translation
            rot[i] = T_t.rotation

        xi_d_world = np.zeros_like(xi_d)
        # body twist x Ad = 속도의 기준점 자체는 body에 있으며 축만 월드 좌표계 기준 정렬
        for i, R in enumerate(rot):
            Ad = np.block([[R, np.zeros((3,3))],
                           [np.zeros((3,3)), R]])
            xi_d_world[i] = Ad @ xi_d[i]

        return t_s, pos, rot, xi_d_world
    
    def raibert_footstep(self, p_hip, v_com, v_des, T_stance, k=0.03):
        '''
        p_hip은 wrapper의 self.L_hip_placement 또는 data.oMf[L_hip].translation에서.
        '''
        return p_hip + 0.5 * T_stance * v_com + k * (v_com - v_des)

    

