import numpy as np
import mujoco
from scipy.spatial import ConvexHull


class Visualizer:
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
            
    def draw_connected_path(self, waypoints, rgba=(0.0, 1.0, 0.0, 1.0), thickness=0.005):
        wp = np.asarray(waypoints)
         # 점이 최소 2개 이상이어야 선을 그을 수 있음
        if len(wp) < 2:
            return

        for i in range(len(wp) - 1):
            if self.scn.ngeom >= self.scn.maxgeom:
                return
        
            p1 = wp[i]
            p2 = wp[i+1]

            # mjv_connector를 사용하여 p1과 p2 사이를 잇는 캡슐 생성
            mujoco.mjv_initGeom(
                self.scn.geoms[self.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), np.zeros(9),
                np.asarray(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(
                self.scn.geoms[self.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE, thickness,
                np.asarray(p1, dtype=np.float64).reshape(3),
                np.asarray(p2, dtype=np.float64).reshape(3),
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

    # 두 점을 잇는 선분 (ConvexHull 테두리 등)
    def draw_line(self, p1, p2, rgba=(0.0, 1.0, 0.0, 1.0), thickness=0.005):
        if self.scn.ngeom >= self.scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), np.zeros(9),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE, thickness,
            np.asarray(p1, dtype=np.float64).reshape(3),
            np.asarray(p2, dtype=np.float64).reshape(3),
        )
        self.scn.ngeom += 1

    # 접촉력 화살표 + 접촉점 구체. p: 접촉점 위치, f: 힘 벡터, scale: m/N
    def draw_contact_force(self, p, f, scale=0.005,
                           rgba=(1.0, 1.0, 0.0, 1.0), shaft=0.006,
                           point_radius=0.008):
        if self.scn.ngeom + 2 > self.scn.maxgeom:
            return
        p = np.asarray(p).reshape(3)
        f = np.asarray(f).reshape(3)
        rgba_f = np.asarray(rgba, dtype=np.float32)

        # 접촉점 구체
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([point_radius, point_radius, point_radius]),
            p, np.eye(3).flatten(), rgba_f,
        )
        self.scn.ngeom += 1

        # 힘 화살표
        end = p + f * scale
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9), rgba_f,
        )
        mujoco.mjv_connector(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW, shaft, p, end,
        )
        self.scn.ngeom += 1

    def draw_geom(self, geom_type, size, pos, mat, rgba):
        """임의 geom을 user_scn에 직접 추가."""
        if self.scn.ngeom >= self.scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            geom_type,
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.asarray(mat, dtype=np.float64).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        self.scn.ngeom += 1
