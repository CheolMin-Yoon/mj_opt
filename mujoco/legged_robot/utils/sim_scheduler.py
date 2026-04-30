import time
import mujoco
import mujoco.viewer

from .visualizer import Visualizer


class SimScheduler:
    """
    MuJoCo viewer 시뮬 루프 + multi-rate 스케줄링.
    - physics: model.opt.timestep (예: 0.002s)
    - control: ctrl_hz로 user callback 호출
    - render: render_hz로 viewer.sync + overlay callback 호출
    - real-time pacing 자동
    """
    def __init__(self, model, data, ctrl_hz=500, render_hz=60):
        self.model, self.data = model, data
        self.ctrl_dt = 1.0 / ctrl_hz
        self.render_dt = 1.0 / render_hz
        self.physics_dt = model.opt.timestep
        self.sim_time = 0.0

    def run(self, on_control, on_render=None, duration=None):
        """
        on_control(sim_time) -> None: data.ctrl 세팅하는 콜백
        on_render(sim_time, visu) -> None: 오버레이 그리기 콜백
        duration: None이면 viewer 닫을 때까지
        """
        with mujoco.viewer.launch_passive(self.model, self.data,
                                          show_left_ui=True,
                                          show_right_ui=True) as viewer:
            view = Visualizer(viewer)
            next_ctrl = next_render = self.sim_time
            wall_start = time.perf_counter()

            while viewer.is_running():
                if duration is not None and self.sim_time >= duration:
                    break
                while self.sim_time < next_render and viewer.is_running():
                    if self.sim_time >= next_ctrl:
                        on_control(self.sim_time)
                        next_ctrl += self.ctrl_dt
                    mujoco.mj_step(self.model, self.data)
                    self.sim_time += self.physics_dt

                view.reset()
                if on_render is not None:
                    on_render(self.sim_time, view)
                viewer.sync()
                next_render += self.render_dt

                slack = (wall_start + self.sim_time) - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
