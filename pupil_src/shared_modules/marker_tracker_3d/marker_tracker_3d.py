"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from marker_tracker_3d.camera_localization_controller import (
    CameraLocalizationController,
)
from marker_tracker_3d.controller import Controller
from marker_tracker_3d.controller_storage import ControllerStorage
from marker_tracker_3d.optimization.model_optimization_controller import (
    ModelOptimizationController,
)
from marker_tracker_3d.optimization.model_optimization_storage import (
    ModelOptimizationStorage,
)
from marker_tracker_3d.user_interface import UserInterface
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
        self._setup_controllers()
        self._setup_ui()

    def _setup_storages(self):
        self._controller_storage = ControllerStorage(
            self._min_marker_perimeter, save_path=self.g_pool.user_dir
        )
        self._model_optimization_storage = ModelOptimizationStorage(
            save_path=self.g_pool.user_dir
        )

    def _setup_controllers(self):
        self._model_optimization_controller = ModelOptimizationController(
            self._model_optimization_storage,
            camera_model=self.g_pool.capture.intrinsics,
            task_manager=self._task_manager,
        )

        self._camera_localization_controller = CameraLocalizationController(
            camera_model=self.g_pool.capture.intrinsics
        )
        self._controller = Controller(
            self._model_optimization_controller,
            self._model_optimization_storage,
            self._camera_localization_controller,
            self._controller_storage,
            plugin=self,
        )

    def _setup_ui(self):
        self._ui = UserInterface(
            self,
            self.g_pool.capture.intrinsics,
            self._model_optimization_controller,
            self._model_optimization_storage,
            self._controller,
            self._controller_storage,
        )

    def get_init_dict(self):
        return self._controller_storage.get_init_dict()
