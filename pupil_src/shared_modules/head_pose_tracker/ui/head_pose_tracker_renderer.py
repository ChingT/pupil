"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections

import numpy as np

from head_pose_tracker import ui as plugin_ui
from head_pose_tracker.ui import gl_renderer_utils as utils


class HeadPoseTrackerRenderer(plugin_ui.GLWindow):
    def __init__(
        self,
        marker_location_storage,
        markers_3d_model_storage,
        camera_localizer_storage,
        camera_intrinsics,
        plugin,
        get_current_frame_index,
    ):
        super().__init__(plugin)

        self._marker_location_storage = marker_location_storage
        self._camera_intrinsics = camera_intrinsics
        self._plugin = plugin
        self._get_current_frame_index = get_current_frame_index

        self._camera_localizer = camera_localizer_storage.item
        self._markers_3d_model = markers_3d_model_storage.item

        self.recent_camera_trace = collections.deque(maxlen=300)

    def _render(self):
        if not self._markers_3d_model.result:
            return

        rotate_center_matrix = self._get_rotate_center_matrix()
        self._render_origin(rotate_center_matrix)

        marker_id_to_points_3d = self._get_marker_id_to_points_3d()
        current_markers = self._get_current_markers()
        self._render_markers(marker_id_to_points_3d, current_markers)

        camera_pose_matrix = self._get_camera_pose_matrix()
        self._render_camera(camera_pose_matrix)

    def _get_rotate_center_matrix(self):
        rotate_center_matrix = np.eye(4, dtype=np.float32)
        rotate_center_matrix[0:3, 3] = -np.array(
            self._markers_3d_model.result["centroid"]
        )
        return rotate_center_matrix

    @staticmethod
    def _render_origin(rotate_center_matrix):
        utils.render_centroid(color=(0.2, 0.2, 0.2, 0.1))
        utils.set_rotate_center(rotate_center_matrix)
        utils.render_coordinate()

    def _get_marker_id_to_points_3d(self):
        return self._markers_3d_model.result["marker_id_to_points_3d"]

    def _get_current_markers(self):
        current_index = self._get_current_frame_index()
        try:
            return self._marker_location_storage[current_index].marker_detection
        except AttributeError:
            return {}

    def _render_markers(self, marker_id_to_points_3d, current_markers):
        for marker_id, points_3d in marker_id_to_points_3d.items():
            color = (1, 0, 0, 0.2) if marker_id in current_markers else (1, 0, 0, 0.1)
            utils.render_polygon_in_3d_window(points_3d, color)

            if self._markers_3d_model.show_marker_id:
                color = (1, 0, 0, 1)
                utils.render_text_in_3d_window(str(marker_id), points_3d[0], color)

    def _get_camera_pose_matrix(self):
        current_frame_index = self._get_current_frame_index()
        ts = self._plugin.g_pool.timestamps[current_frame_index]
        try:
            pose_datum = self._camera_localizer.pose_bisector.by_ts(ts)
        except ValueError:
            camera_trace = np.full((3,), np.nan)
            camera_pose_matrix = None
        else:
            camera_trace = pose_datum["camera_trace"]
            camera_pose_matrix = pose_datum["camera_pose_matrix"]

        # recent_camera_trace is updated no matter show_camera_trace is on or not
        self.recent_camera_trace.append(camera_trace)

        return camera_pose_matrix

    def _render_camera(self, camera_pose_matrix):
        color = (0.2, 0.2, 0.2, 0.1)
        if self._camera_localizer.show_camera_trace:
            utils.render_camera_trace(self.recent_camera_trace, color)

        if camera_pose_matrix is not None:
            utils.render_camera_frustum(
                camera_pose_matrix, self._camera_intrinsics, color
            )
