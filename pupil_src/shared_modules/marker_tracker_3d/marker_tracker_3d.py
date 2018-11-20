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

import marker_tracker_3d.math
import marker_tracker_3d.utils
from marker_tracker_3d import optimization
from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.camera_model import CameraModel
from marker_tracker_3d.marker_detector import MarkerDetector
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

        self.marker_detector = MarkerDetector(self.storage, min_marker_perimeter)
        self.ui = UserInterface(self, self.storage)
        self.camera_model = CameraModel()
        self.optimization_controller = optimization.Controller(
            self.storage, self.ui.update_menu
        )

        # for tracking
        self.min_number_of_markers_per_frame_for_loc = 2
        self.marker_model_3d = CameraLocalizer()

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
        self.optimization_controller.cleanup()

    def restart(self):
        self.storage.reset()
        self.ui.update_menu()
        self.optimization_controller.restart()

    def save_data(self):
        self.optimization_controller.save_data()

    def get_init_dict(self):
        d = super().get_init_dict()
        d["min_marker_perimeter"] = self.marker_detector.min_marker_perimeter

        return d

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            self.early_exit()
            return

        self.marker_detector.detect(frame)
        self.optimization_controller.fetch_extrinsics()

        if len(self.storage.markers) < self.min_number_of_markers_per_frame_for_loc:
            self.early_exit()
            return

        if self.marker_model_3d is not None:
            self.update_camera_extrinsics()  # TODO: move to controller

        self.optimization_controller.send_marker_data()

    # TODO by now this is doing so little, maybe we should rename it to
    # pop_camera_trace or something similar
    def early_exit(self):
        if len(self.storage.camera_trace):
            self.storage.camera_trace.popleft()

    def update_camera_extrinsics(self):
        self.storage.camera_extrinsics = self.marker_model_3d.current_camera(
            self.storage.markers, self.storage.previous_camera_extrinsics
        )
        if self.storage.camera_extrinsics is None:
            # Do not set previous_camera_extrinsics to None to ensure a decent initial
            # guess for the next solve_pnp call
            self.storage.camera_trace.append(None)
            self.storage.camera_trace_all.append(None)
        else:
            self.storage.previous_camera_extrinsics = self.storage.camera_extrinsics

            camera_pose_matrix = marker_tracker_3d.math.get_camera_pose_mat(
                self.storage.camera_extrinsics
            )
            self.storage.camera_trace.append(camera_pose_matrix[0:3, 3])
            self.storage.camera_trace_all.append(camera_pose_matrix[0:3, 3])

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
