"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from head_pose_tracker import ui as plugin_ui, controller, storage

from observable import Observable
from plugin import Plugin
from tasklib.manager import PluginTaskManager


class Online_Head_Pose_Tracker(Plugin, Observable):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the
    environment.
    """

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self._task_manager = PluginTaskManager(plugin=self)

        self._setup_storages()
        self._setup_controllers()
        self._setup_renderers()
        self._setup_menus()

    def _setup_storages(self):
        self._online_settings_storage = storage.OnlineSettingsStorage(
            self.g_pool.user_dir, plugin=self
        )
        self._marker_location_storage = storage.OnlineMarkerLocationStorage()
        self._markers_3d_model_storage = storage.Markers3DModelStorage(
            self.g_pool.user_dir, plugin=self
        )
        self._camera_localizer_storage = storage.OnlineCameraLocalizerStorage()

    def _setup_controllers(self):
        self._controller = controller.OnlineController(
            self._online_settings_storage,
            self._marker_location_storage,
            self._markers_3d_model_storage,
            self._camera_localizer_storage,
            self.g_pool.capture.intrinsics,
            self._task_manager,
            user_dir=self.g_pool.user_dir,
            plugin=self,
        )

    def _setup_renderers(self):
        self._marker_location_renderer = plugin_ui.MarkerLocationRenderer(
            self._online_settings_storage,
            self._marker_location_storage,
            self._markers_3d_model_storage,
            plugin=self,
        )
        self._head_pose_tracker_3d_renderer = plugin_ui.HeadPoseTracker3DRenderer(
            self._online_settings_storage,
            self._marker_location_storage,
            self._markers_3d_model_storage,
            self._camera_localizer_storage,
            self.g_pool.capture.intrinsics,
            plugin=self,
        )

    def _setup_menus(self):
        self._markers_3d_model_menu = plugin_ui.OnlineMarkers3DModelMenu(
            self._controller,
            self._online_settings_storage,
            self._markers_3d_model_storage,
        )
        self._camera_localizer_menu = plugin_ui.OnlineCameraLocalizerMenu(
            self._online_settings_storage, self._camera_localizer_storage
        )
        self._head_pose_tracker_menu = plugin_ui.OnlineHeadPoseTrackerMenu(
            self._markers_3d_model_menu,
            self._camera_localizer_menu,
            self._head_pose_tracker_3d_renderer,
            plugin=self,
        )
