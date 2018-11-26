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
from plugin import Plugin

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


class Marker_Tracker_3D(Plugin):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the environment.
    """

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, min_marker_perimeter=100):
        super().__init__(g_pool)

        self.storage = Storage()

        self.ui = UserInterface(self, self.storage)

        self.controller = Controller(
            self.storage,
            self.g_pool.capture.intrinsics,
            self.ui.update_menu,
            min_marker_perimeter,
        )

        # for experiments
        self.robustness = list()
        self.all_frames = list()
        self.reprojection_errors = list()

    def init_ui(self):
        self.ui.init_ui()

    def deinit_ui(self):
        self.ui.deinit_ui()

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """

        self.ui.close_window()
        self.controller.cleanup()

    def restart(self):
        self.storage.reset()
        self.ui.update_menu()
        self.controller.restart()

    def save_data(self):
        self.controller.save_data()

    def get_init_dict(self):
        d = super().get_init_dict()
        d["min_marker_perimeter"] = self.controller.marker_detector.min_marker_perimeter
        return d

    def recent_events(self, events):
        frame = events.get("frame")
        self.controller.recent_events(frame)

    def gl_display(self):
        self.ui.gl_display(
            self.g_pool.capture.intrinsics.K, self.g_pool.capture.intrinsics.resolution
        )

    def on_resize(self, window, w, h):
        self.ui.on_resize(window, w, h)

    def on_window_key(self, window, key, scancode, action, mods):
        self.ui.on_window_key(window, key, scancode, action, mods)

    def on_close(self, window=None):
        self.ui.on_close(window)

    def on_window_mouse_button(self, window, button, action, mods):
        self.ui.on_window_mouse_button(window, button, action, mods)

    def on_window_pos(self, window, x, y):
        self.ui.on_window_pos(window, x, y)

    def on_scroll(self, window, x, y):
        self.ui.on_scroll(window, x, y)
