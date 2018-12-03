"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

from marker_tracker_3d.controller import Controller
from marker_tracker_3d.storage import Storage
from marker_tracker_3d.user_interface import UserInterface
from observable import Observable
from plugin import Plugin

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


class Marker_Tracker_3D(Plugin, Observable):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the environment.
    """

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, min_marker_perimeter=100):
        super().__init__(g_pool)

        self.storage = Storage()

        self.controller = Controller(
            self, self.storage, self.g_pool.capture.intrinsics, min_marker_perimeter
        )
        self.ui = UserInterface(self, self.storage, self.g_pool.capture.intrinsics)

    def get_init_dict(self):
        d = {
            "min_marker_perimeter": self.controller.marker_detector.min_marker_perimeter
        }
        return d
