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
    

    def build_foot_trajectories(self, footsteps, init_lf, init_rf,
                                step_time, dsp_time, init_dsp_extra,
                                dt, step_height):
        """
        DCM footstep 시퀀스에 맞춰 좌/우 발 (N,3) 위치/속도/가속도 배열을 사전 계산.
        SSP 구간은 compute_bezier_trajectory 사용 → 가속도까지 연속.

        ssp_time(=step_time-dsp_time)이 0 이하면 ZeroDivision → assert.

        Parameters
        ----------
        footsteps        : List[(x, y)] 발자국 위치 (그대로 사용)
        init_lf, init_rf : (3,) 초기 발 위치 (world)
        step_time        : 한 스텝 총 시간
        dsp_time         : 한 스텝 내 DSP(양발지지) 시간
        init_dsp_extra   : 첫 스텝의 추가 DSP
        dt               : 샘플 주기
        step_height      : swing 최고 높이

        Returns
        -------
        l_pos, r_pos : (N, 3) world 위치
        l_vel, r_vel : (N, 3) world 속도
        l_acc, r_acc : (N, 3) world 가속도
        """
        assert step_time - dsp_time > 0, \
            f"ssp_time must be > 0, got step_time={step_time}, dsp_time={dsp_time}"

        n_steps = len(footsteps)

        def step_t(i):  return step_time + (init_dsp_extra if i == 0 else 0.0)
        def dsp_t(i):   return dsp_time  + (init_dsp_extra if i == 0 else 0.0)
        def n_samp(i):  return int(step_t(i) / dt)

        N_total = sum(n_samp(i) for i in range(n_steps))
        l_pos = np.zeros((N_total, 3));  r_pos = np.zeros((N_total, 3))
        l_vel = np.zeros((N_total, 3));  r_vel = np.zeros((N_total, 3))
        l_acc = np.zeros((N_total, 3));  r_acc = np.zeros((N_total, 3))

        # 현재 발 위치 (stance 발의 이전 착지점)
        l_now, r_now = init_lf.copy(), init_rf.copy()
        ground_l, ground_r = init_lf[2], init_rf[2]

        idx_offset = 0
        for i in range(n_steps):
            ns    = n_samp(i)
            dsp_i = dsp_t(i)
            ssp_i = step_t(i) - dsp_i
            is_right_swing = (i % 2 == 0)

            # swing target = footsteps[i+1] (footsteps의 x, y 그대로 사용)
            if i + 1 < n_steps:
                fx, fy = footsteps[i + 1]
                swing_target = np.array([fx, fy,
                                         ground_r if is_right_swing else ground_l])
            else:
                swing_target = None
            swing_start = r_now if is_right_swing else l_now

            for k in range(ns):
                t_local = k * dt
                idx = idx_offset + k

                if t_local < dsp_i or swing_target is None:
                    # DSP 또는 마지막 스텝: 양발 정지
                    l_pos[idx] = l_now;  r_pos[idx] = r_now
                    # vel/acc는 0 (np.zeros 기본값)
                    continue

                phase = np.clip((t_local - dsp_i) / ssp_i, 0.0, 1.0)
                p, v, a = self.compute_bezier_trajectory(
                    phase, swing_start, swing_target, ssp_i, step_height
                )

                if is_right_swing:
                    r_pos[idx], r_vel[idx], r_acc[idx] = p, v, a
                    l_pos[idx] = l_now
                else:
                    l_pos[idx], l_vel[idx], l_acc[idx] = p, v, a
                    r_pos[idx] = r_now

            # 다음 스텝 시작 시 swing 발이 새 위치에 착지
            if swing_target is not None:
                if is_right_swing:
                    r_now = swing_target.copy()
                else:
                    l_now = swing_target.copy()
            idx_offset += ns

        return l_pos, r_pos, l_vel, r_vel, l_acc, r_acc

    

