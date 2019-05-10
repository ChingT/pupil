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

from camera_extrinsics_measurer import ui as plugin_ui, function
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

        # self._render_markers()
        self._render_camera()

    def _render_markers(self):
        marker_id_to_points_3d = self._optimization_storage.marker_id_to_points_3d

        for marker_id, points_3d in marker_id_to_points_3d.items():
            color = (1, 0, 0, 0.2)
            utils.render_polygon_in_3d_window(points_3d, color)

    def _render_camera(self):
        current_poses = self._localization_storage.current_pose
        color = (0.2, 0.2, 0.2, 0.1)

        for camera_name in function.utils.camera_name:
            camera_pose_matrix = current_poses[camera_name]["camera_pose_matrix"]
            if camera_pose_matrix is not None:
                utils.set_rotate_center(np.eye(4, dtype=np.float32))

                utils.render_camera_frustum(
                    camera_pose_matrix, self._intrinsics_dict[camera_name], color
                )
