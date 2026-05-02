# A Benchmarking of DCM Based Architectures for Position and Velocity Controlled Walking of Humanoid Robots
# DCM Trajectory Planner 저자가 설명하는 첫번째 Layer

# 이 레이어에서는 

import numpy as np
from typing import List, Tuple


class TrajectoryOptimization:

    def __init__(
        self,
        z_c: float,
        g: float = 9.81,
        step_time: float = 1.0,
        dsp_time: float = 0.1,
        dt: float = 0.001,
        init_dsp_extra: float = 0.12,
    ):
        self.z_c = z_c
        self.omega = np.sqrt(g / z_c)
        self.step_time = step_time
        self.dsp_time = dsp_time
        self.ssp_time = step_time - dsp_time
        self.dt = dt
        self.samples_per_step = int(step_time / dt)
        self.init_dsp_extra = init_dsp_extra  # 첫 스텝 DSP 추가 시간

    # ========================================================================= #
    # Helper: 스텝별 시간/샘플 수 (첫 스텝만 DSP 확장)
    # ========================================================================= #
    def _step_time_for(self, i: int) -> float:
        """i번째 스텝의 총 시간"""
        if i == 0 and self.init_dsp_extra > 0:
            return self.step_time + self.init_dsp_extra
        return self.step_time

    def _dsp_time_for(self, i: int) -> float:
        """i번째 스텝의 DSP 시간"""
        if i == 0 and self.init_dsp_extra > 0:
            return self.dsp_time + self.init_dsp_extra
        return self.dsp_time

    def _samples_for(self, i: int) -> int:
        return int(self._step_time_for(i) / self.dt)

    def _total_samples(self, n_steps: int) -> int:
        return sum(self._samples_for(i) for i in range(n_steps))

    def _step_start_idx(self, step: int) -> int:
        return sum(self._samples_for(i) for i in range(step))

    # ========================================================================= #
    # 1. Footstep Plan
    # ========================================================================= #
    def plan_footsteps(
        self,
        n_steps: int,
        step_length: float,
        step_width: float,
        init_xy: np.ndarray = np.array([0.035, 0.0]) # 근데 이게 아마 실제로 0.0351, 0.0 일껄
    ) -> List[Tuple[float, float]]:                  # init_xy는 초기 CoM
        
        # 리스트로 저장
        footsteps = []
        
        # 첫발은 왼발부터 시작
        for i in range(n_steps):
            if i == 0:
                x = init_xy[0]
                y = step_width  # sway가 CoM을 왼발(지지발)로 이동시키므로 정확히 step_width
            else:
                x = init_xy[0] + i * step_length
                
                if i % 2 != 0:
                    y = -step_width   # 오른발
                else:
                    y = step_width  # 왼발
            footsteps.append((x,y))
        return footsteps

    # ========================================================================= #
    # 2. DCM Trajectory
    # ========================================================================= #
    def compute_dcm_trajectory(
        self,
        footsteps: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = len(footsteps)
        total_samples = self._total_samples(n_steps)

        ref_dcm = np.zeros((total_samples, 2))
        ref_dcm_vel = np.zeros((total_samples, 2))

        # DCM end-of-step (역방향 계산) — 각 스텝의 step_time 사용
        dcm_eos = np.zeros((n_steps, 2))
        dcm_eos[-1] = np.array(footsteps[-1])

        for i in range(n_steps - 2, -1, -1):
            next_zmp = np.array(footsteps[i + 1])
            exp_neg = np.exp(-self.omega * self._step_time_for(i + 1))
            dcm_eos[i] = next_zmp + (dcm_eos[i + 1] - next_zmp) * exp_neg

        # DCM 순방향 생성
        for i in range(n_steps):
            start_idx = self._step_start_idx(i)
            samples_i = self._samples_for(i)
            step_time_i = self._step_time_for(i)
            current_zmp = np.array(footsteps[i])
            xi_end = dcm_eos[i]

            for k in range(samples_i):
                idx = start_idx + k
                t = k * self.dt
                t_remaining = step_time_i - t

                current_dcm = current_zmp + (xi_end - current_zmp) * np.exp(-self.omega * t_remaining)
                ref_dcm[idx] = current_dcm
                ref_dcm_vel[idx] = self.omega * (current_dcm - current_zmp)

        return ref_dcm, ref_dcm_vel

    # ========================================================================= #
    # 3. CoM Trajectory  (DCM 1차 ODE 적분: ẋ = ω(ξ - x))
    #    DCM → CoM 변환은 stable 1차 시스템 (ω>0인 1차 저역통과).
    #    한 step 내부에서 ξ가 exponential이지만 호출 시 ref_dcm[k]를 고정하고
    #    forward Euler로 적분 — exact 해를 step-wise로 풀려면 cosh/sinh가
    #    들어가는데 unstable mode 동반 → 수치적으로 forward Euler가 더 안전.
    # ========================================================================= #
    def compute_com_trajectory(
        self,
        ref_dcm: np.ndarray,
        init_com_xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(ref_dcm)
        ref_com_pos = np.zeros((N, 2))
        ref_com_vel = np.zeros((N, 2))

        x = init_com_xy[:2].copy()
        for k in range(N):
            ref_com_pos[k] = x
            dx = self.omega * (ref_dcm[k] - x)   # LIPM 정의
            ref_com_vel[k] = dx
            x = x + dx * self.dt                 # forward Euler (1차 stable system)

        return ref_com_pos, ref_com_vel

    # ========================================================================= #
    # Main Wrapper — CoM/DCM/footstep만 담당 (발 궤적은 motion_planner 사용)
    # ========================================================================= #
    def compute_all_trajectories(
        self,
        n_steps: int,
        step_length: float,
        step_width: float,
        init_com: np.ndarray,     # (3,) 월드 좌표 CoM
    ):
        # 1. 발자국 계획
        footsteps = self.plan_footsteps(n_steps, step_length, step_width, init_xy=init_com[:2])

        # 2. DCM 궤적
        ref_dcm, ref_dcm_vel = self.compute_dcm_trajectory(footsteps)

        # 3. CoM 궤적
        com_xy, com_vel = self.compute_com_trajectory(ref_dcm, init_com[:2])
        com_pos = np.column_stack([com_xy, np.full(len(com_xy), init_com[2])])

        return footsteps, ref_dcm, ref_dcm_vel, com_pos, com_vel