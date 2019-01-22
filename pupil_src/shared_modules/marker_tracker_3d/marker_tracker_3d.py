"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from marker_tracker_3d import ui as plugin_ui, controller, storage
from observable import Observable
from plugin import Plugin
from tasklib.manager import PluginTaskManager


class Marker_Tracker_3D(Plugin, Observable):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the
    environment.
    """

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, min_marker_perimeter=100):
        super().__init__(g_pool)
        self._min_marker_perimeter = min_marker_perimeter
        self._task_manager = PluginTaskManager(plugin=self)

        self._setup_storages()
        self._setup_controller()
        self._setup_ui()

    def _setup_storages(self):
        self._controller_storage = storage.ControllerStorage(
            self._min_marker_perimeter, save_path=self.g_pool.user_dir
        )
        self._model_storage = storage.ModelStorage(save_path=self.g_pool.user_dir)

    def _setup_controller(self):
        self._controller = controller.Controller(
            controller_storage=self._controller_storage,
            model_storage=self._model_storage,
            camera_intrinsics=self.g_pool.capture.intrinsics,
            task_manager=self._task_manager,
            plugin=self,
        )

    def _setup_ui(self):
        self._ui = plugin_ui.UserInterface(
            self,
            self.g_pool.capture.intrinsics,
            self._controller,
            self._controller_storage,
            self._model_storage,
        )

    def get_init_dict(self):
        return self._controller_storage.get_init_dict()
