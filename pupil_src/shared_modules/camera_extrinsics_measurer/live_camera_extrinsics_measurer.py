"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import time

from camera_extrinsics_measurer import camera_names
from camera_extrinsics_measurer import ui as plugin_ui, controller, storage


class Live_Camera_Extrinsics_Measurer:
    def __init__(
        self,
        plmodel_dir,
        intrinsics_dict,
        optimize_markers_3d_model=False,
        optimize_camera_intrinsics=True,
        render_markers_in_3d_window=1,
        convert_to_cam_coordinate=1,
        show_marker_id_in_3d_window=False,
        window_size=(1000, 1000),
        window_position=(0, 0),
    ):
        self._plmodel_dir = plmodel_dir
        self._intrinsics_dict = intrinsics_dict
        self._online_settings = storage.OnlineSettings(
            (
                optimize_markers_3d_model,
                optimize_camera_intrinsics,
                render_markers_in_3d_window,
                convert_to_cam_coordinate,
                show_marker_id_in_3d_window,
                window_size,
                window_position,
            )
        )
        self._setup_storages()
        self._setup_controllers()
        self._setup_renderers()

        self._last_time = {name: 0 for name in camera_names}

    def recent_events(self, frame, camera_name):
        now = time.time()
        if now - self._last_time[camera_name] < 1 / 30:
            return

        self._last_time[camera_name] = now

        self._controller.recent_events(frame, camera_name)

        if camera_name == "world":
            self._live_camera_extrinsics_measurer_3d_renderer.on_gl_display()

    def calculate_markers_3d_model(self, camera_name):
        self._controller.calculate_markers_3d_model(camera_name)

    def _setup_storages(self):
        self._detection_storage = storage.OnlineDetectionStorage()
        self._optimization_storage = storage.LiveOptimizationStorage(self._plmodel_dir)
        self._localization_storage = storage.OnlineLocalizationStorage()

    def _setup_controllers(self):
        self._controller = controller.LiveController(
            self._online_settings,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            self._intrinsics_dict,
            self._plmodel_dir,
        )

    def _setup_renderers(self):
        self._live_camera_extrinsics_measurer_3d_renderer = plugin_ui.LiveCameraExtrinsicsMeasurer3dRenderer(
            self._online_settings,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            self._intrinsics_dict,
        )
