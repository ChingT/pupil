"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from head_pose_tracker import ui as plugin_ui, controller, model
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

    def __init__(self, g_pool, predetermined_origin_marker_id=None):
        super().__init__(g_pool)
        self._task_manager = PluginTaskManager(plugin=self)

        self._setup_storages(predetermined_origin_marker_id)
        self._setup_controller()
        self._setup_ui()

    def _setup_storages(self, predetermined_origin_marker_id):
        self._controller_storage = model.ControllerStorage(
            save_path=self.g_pool.user_dir
        )
        self._model_storage = model.ModelStorage(
            predetermined_origin_marker_id, save_path=self.g_pool.user_dir
        )

    def _setup_controller(self):
        self._general_controller = controller.OnlineGeneralController(
            self._controller_storage,
            self._model_storage,
            self.g_pool.capture.intrinsics,
            self._task_manager,
            plugin=self,
            save_path=self.g_pool.user_dir,
        )

    def _setup_ui(self):
        self._head_pose_tracker_menu = plugin_ui.OnlineHeadPoseTrackerMenu(
            self._general_controller,
            self._controller_storage,
            self._model_storage,
            plugin=self,
        )
        self.visualization_3d_window = plugin_ui.Visualization3dWindow(
            self.g_pool.capture.intrinsics,
            self._controller_storage,
            self._model_storage,
            plugin=self,
        )
        self._marker_renderer = plugin_ui.MarkerRenderer(
            self._controller_storage, self._model_storage, plugin=self
        )
