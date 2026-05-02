import numpy as np
import pinocchio as pin


class TrajectoryGenerator:
    def __init__(self, wp_list, R_list, target_speed):
        self.wp_list = wp_list   # N, 3
        self.R_list = R_list     # N, 3, 3
        self.N = len(wp_list)
        
        self.times = self.compute_time_wp_to_wp(target_speed)
        self.vel, self.acc = self.compute_hueristic()
        self.segments_coeffs = []
        for i in range(self.N - 1):
            T_segment = self.times[i+1] - self.times[i] # 구간 소요 시간
            coeffs = self.compute_segment_coeffs(
                self.wp_list[i], self.vel[i], self.acc[i],
                self.wp_list[i+1], self.vel[i+1], self.acc[i+1],
                T_segment
            )
            self.segments_coeffs.append(coeffs)

    # 중앙 차분 시 wp간 도달 시간이 거리에 따라 다름 따라서 전체 목표 속도를 일정하게 유지하기 위해
    # 각 wp의 도달 시간을 유클라디안 노름으로 계산
    def compute_time_wp_to_wp(self, target_speed):
        times = np.zeros(self.N)
        for i in range(1, self.N):
            distance = np.linalg.norm(self.wp_list[i] - self.wp_list[i-1])
            dt = max(distance / target_speed, 0.005)
            times[i] = times[i-1] + dt
        return times    
    
    # wp 리스트와 국소 도달 시간 T로 각 wp의 vf, af 계산 이때 2차 정확도를 가진 중앙 차분법으로 계산
    # 만약 각진 궤적을 원한다면 for 루프를 주석 처리
    def compute_hueristic(self):
        
        N = self.N  
        vel = np.zeros((N, 3))
        acc = np.zeros((N, 3))
        
        vel[0] = np.zeros(3)
        vel[-1] = np.zeros(3)
        acc[0] = np.zeros(3)
        acc[-1] = np.zeros(3)
        '''
        for i in range(1, self.N-1):
            dt_prev = self.times[i] - self.times[i-1]
            dt_next = self.times[i+1] - self.times[i]
            vel[i] = (self.wp_list[i+1] - self.wp_list[i-1]) / (dt_prev + dt_next)
            
            v_next = (self.wp_list[i+1] - self.wp_list[i]) / dt_next
            v_prev = (self.wp_list[i] - self.wp_list[i-1]) / dt_prev
            acc[i] = (v_next - v_prev) / ((dt_prev + dt_next) / 2)
        '''
        return vel, acc
    
    # wp to wp 간 초기 p0, v0, a0, 도달 시 pf, vf, af, 국소 도달 시간 T로 5차 다항식 계수 계산
    def compute_segment_coeffs(self, p0, v0, a0, pf, vf, af, T):
        
        c0 = p0
        c1 = v0
        c2 = 0.5 * a0
        
        A = np.array([[T**3,   T**4,      T**5],
                      [3*T**2, 4 * T**3,  5 * T**4],
                      [6*T,    12 * T**2, 20 * T**3]])
        
        B = np.array([pf - (c0 + c1 * T + c2 * T**2),
                      vf - (c1 + 2 * c2 * T),
                      af - (2 * c2)])
        
        c3, c4, c5 = np.linalg.solve(A, B)
        
        return np.array([c0, c1, c2, c3, c4, c5])

    def evaluate(self, current_t):
        # 1. 궤적 시간이 끝났으면 마지막 상태 반환
        if current_t >= self.times[-1]:
            return pin.SE3(self.R_list[-1], self.wp_list[-1]), np.zeros(6)
            
        # 2. 현재 시간이 어느 구간(idx)에 속하는지 찾기
        idx = 0
        for i in range(self.N - 1):
            if self.times[i] <= current_t < self.times[i+1]:
                idx = i
                break
                
        # 3. 질문하신 핵심! 국소 시간(dt) 계산
        t_local = current_t - self.times[idx]
        T_segment = self.times[idx+1] - self.times[idx]
        
        # 4. 병진(Translation) 5차 보간 (기존 코드)
        c0, c1, c2, c3, c4, c5 = self.segments_coeffs[idx]
        pos = c0 + c1*t_local + c2*t_local**2 + c3*t_local**3 + c4*t_local**4 + c5*t_local**5
        vel_lin = c1 + 2*c2*t_local + 3*c3*t_local**2 + 4*c4*t_local**3 + 5*c5*t_local**4
    
        
        # 6. 회전을 위한 5차 다항식 스케일링
        tau = t_local / T_segment
        s_rot = 10*tau**3 - 15*tau**4 + 6*tau**5
        # s_dot은 시간(t)에 대한 s_rot의 미분값
        s_dot = (30*tau**2 - 60*tau**3 + 30*tau**4) / T_segment
        
        # 7. 자세 보간
        R0 = self.R_list[idx]
        Rf = self.R_list[idx+1]
        # R_rel: R0에서 Rf로 가기 위한 상대 회전
        R_rel = Rf @ R0.T
        # exp3(log3(R_rel) * s) 를 통해 현재 회전 계산
        current_R = pin.exp3(pin.log3(R_rel) * s_rot) @ R0
        
        # 8. 각속도(Angular Velocity) 계산
        # 세계 좌표계 기준 각속도: omega = log3(R_rel) * s_dot
        # (주의: pin.log3 결과는 [wx, wy, wz] 벡터임)
        vel_ang = pin.log3(R_rel) * s_dot 
        
        # ... (생략: pos, vel_lin 계산)
        interpolated_oM_ee = pin.SE3(current_R, pos)
        twist = np.concatenate([vel_lin, vel_ang])
        
        return interpolated_oM_ee, twist