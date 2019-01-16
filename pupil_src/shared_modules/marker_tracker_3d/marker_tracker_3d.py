"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from marker_tracker_3d.controller import Controller
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

        self._task_manager = PluginTaskManager(plugin=self)
        self.controller = Controller(
            self.g_pool.capture.intrinsics,
            min_marker_perimeter=min_marker_perimeter,
            task_manager=self._task_manager,
            plugin=self,
        )
        self.ui = UserInterface(self, self.g_pool.capture.intrinsics)

    def get_init_dict(self):
        return self.controller.get_init_dict()
