import numpy as np
import mujoco


class ViewerOverlay:
    """
    mujoco viewer.user_scn에 디버그용 시각화 (좌표계, 궤적, 접촉력)을 그린다.
    매 렌더 프레임 시작 시 reset() 호출 후 draw_* 호출.
    """

    _AXIS_COLORS = (
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),  # X 빨강
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),  # Y 초록
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),  # Z 파랑
    )

    def __init__(self, viewer):
        self.viewer = viewer
        self.scn = viewer.user_scn

    # 매 렌더 프레임 시작에 호출
    def reset(self):
        self.scn.ngeom = 0

    # 단일 프레임 좌표계 그리기 (XYZ 화살표)
    def draw_axes(self, pos, R, size=0.1, shaft=0.005):
        R = np.asarray(R).reshape(3, 3)
        pos = np.asarray(pos).reshape(3)
        for i in range(3):
            if self.scn.ngeom >= self.scn.maxgeom:
                return
            end = pos + R[:, i] * size
            mujoco.mjv_initGeom(
                self.scn.geoms[self.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3), np.zeros(3), np.zeros(9),
                self._AXIS_COLORS[i],
            )
            mujoco.mjv_connector(
                self.scn.geoms[self.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW, shaft,
                pos, end,
            )
            self.scn.ngeom += 1

    # 웨이포인트/궤적 점 시각화
    def draw_trajectory(self, waypoints, radius=0.01,
                        rgba=(1.0, 0.5, 0.0, 1.0)):
        wp = np.asarray(waypoints)
        if wp.ndim == 1:
            wp = wp.reshape(1, 3)
        rgba_f = np.asarray(rgba, dtype=np.float32)
        size = np.array([radius, radius, radius])
        for p in wp:
            if self.scn.ngeom >= self.scn.maxgeom:
                return
            mujoco.mjv_initGeom(
                self.scn.geoms[self.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size, np.asarray(p).reshape(3), np.eye(3).flatten(),
                rgba_f,
            )
            self.scn.ngeom += 1

    # 여러 EE 좌표계 한꺼번에. oMf_dict: {key: pin.SE3 또는 (pos, R) 튜플}
    def draw_frame_set(self, oMf_dict, size=0.1):
        for _, oMf in oMf_dict.items():
            if hasattr(oMf, 'translation') and hasattr(oMf, 'rotation'):
                self.draw_axes(oMf.translation, oMf.rotation, size=size)
            else:
                pos, R = oMf
                self.draw_axes(pos, R, size=size)

    # 접촉력 화살표. p: 접촉점 위치, f: 힘 벡터, scale: m/N
    def draw_contact_force(self, p, f, scale=0.005,
                           rgba=(1.0, 1.0, 0.0, 1.0), shaft=0.006):
        if self.scn.ngeom >= self.scn.maxgeom:
            return
        p = np.asarray(p).reshape(3)
        f = np.asarray(f).reshape(3)
        end = p + f * scale
        rgba_f = np.asarray(rgba, dtype=np.float32)
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9),
            rgba_f,
        )
        mujoco.mjv_connector(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW, shaft,
            p, end,
        )
        self.scn.ngeom += 1
