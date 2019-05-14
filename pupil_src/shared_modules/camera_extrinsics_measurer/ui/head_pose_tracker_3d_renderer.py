"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import numpy as np

from camera_extrinsics_measurer import ui as plugin_ui, function, camera_names
from camera_extrinsics_measurer.ui import gl_renderer_utils as utils


class HeadPoseTracker3DRenderer(plugin_ui.GLWindow):
    def __init__(
        self,
        general_settings,
        detection_storage,
        optimization_storage,
        localization_storage,
        intrinsics_dict,
        plugin,
    ):
        super().__init__(general_settings, plugin)

        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._optimization_storage = optimization_storage
        self._localization_storage = localization_storage
        self._intrinsics_dict = intrinsics_dict

    def _render(self):
        if not self._optimization_storage.calculated:
            return

        self._render_origin()
        self._render_markers()
        self._render_camera()

    def _render_origin(self):
        utils.render_centroid(color=(0.2, 0.2, 0.2, 0.1))
        self._set_rotate_center_matrix()
        utils.render_coordinate()

    def _set_rotate_center_matrix(self):
        rotate_center_matrix = np.eye(4, dtype=np.float32)
        if not self._general_settings.convert_to_world_coordinate:
            rotate_center_matrix[0:3, 3] = -self._optimization_storage.centroid
        utils.set_rotate_center(rotate_center_matrix)

    def _render_markers(self):
        if not self._general_settings.render_markers_in_3d_window:
            return

        if self._general_settings.convert_to_world_coordinate:
            current_pose_world = self._localization_storage.current_pose["world"]
            camera_extrinsics = current_pose_world["camera_extrinsics"]
            if camera_extrinsics is None:
                return

            inv = function.utils.convert_extrinsic_to_matrix(camera_extrinsics)
            utils.shift_render_center(inv)

        marker_id_to_points_3d = self._optimization_storage.marker_id_to_points_3d
        for marker_id, points_3d in marker_id_to_points_3d.items():
            color = (1, 0, 0, 0.2)
            utils.render_polygon_in_3d_window(points_3d, color)

            if self._general_settings.show_marker_id_in_3d_window:
                color = (1, 0, 0, 1)
                utils.render_text_in_3d_window(str(marker_id), points_3d[0], color)

    def _render_camera(self):
        if self._general_settings.convert_to_world_coordinate:
            current_poses = self._localization_storage.current_pose_converted
        else:
            current_poses = self._localization_storage.current_pose

        color = (0.2, 0.2, 0.2, 0.1)

        for camera_name in camera_names:
            try:
                camera_pose_matrix = current_poses[camera_name]["camera_pose_matrix"]
            except KeyError:
                continue

            if camera_pose_matrix is not None:
                self._set_rotate_center_matrix()

                utils.render_camera_frustum(
                    camera_pose_matrix, self._intrinsics_dict[camera_name], color
                )
