import numpy as np


class GaitScheduler:
    def __init__(self, gait_hz: float, duty: float):
        self.gait_period  = 1.0 / gait_hz
        self.gait_duty    = duty
        self.stance_time  = duty * self.gait_period
        self.swing_time   = (1.0 - duty) * self.gait_period
        self.phase_offset = np.array([0.,0.5])
    
    '''
    현재 시간 t에 따른 각 다리의 상태를 반환하는 함수
    
    입력:
    
    출력:
    
    '''
    def get_gait_state(self, t: float):
        phases = np.mod(t / self.gait_period + self.phase_offset, 1.0)
        contact_mask = (phases < self.gait_duty).astype(int)
        
        swing_phases = np.zeros_like(phases)
        for i in range(len(phases)):
            if contact_mask[i] == 0:
                swing_phases[i] = (phases[i] - self.gait_duty) / (1.0 - self.gait_duty)
                
        return contact_mask, swing_phases

    
    
        