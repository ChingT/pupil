import logging

import OpenGL.GL as gl
import cv2
import numpy as np
from pyglui.cygl import utils as cygl_utils

import square_marker_detect
from marker_tracker_3d import ui as plugin_ui

logger = logging.getLogger(__name__)


class UserInterface:
    def __init__(
        self, plugin, intrinsics, controller, controller_storage, model_storage
    ):
        self._plugin = plugin

        self._controller = controller
        self._controller_storage = controller_storage

        self._head_pose_tracker_menu = plugin_ui.HeadPoseTrackerMenu(
            controller_storage, model_storage
        )
        self._visualization_3d_window = plugin_ui.Visualization3dWindow(
            intrinsics, controller_storage, model_storage
        )

        self._plugin.add_observer("init_ui", self._on_init_ui)
        self._plugin.add_observer("deinit_ui", self._on_deinit_ui)
        self._plugin.add_observer("gl_display", self._on_gl_display)
        self._plugin.add_observer("cleanup", self._on_cleanup)

        self._head_pose_tracker_menu.add_observer(
            "on_open_3d_window", self._visualization_3d_window.on_open_window
        )
        self._head_pose_tracker_menu.add_observer(
            "on_close_3d_window", self._visualization_3d_window.on_close_window
        )
        self._head_pose_tracker_menu.add_observer(
            "on_reset_button_click", self._on_reset_button_click
        )
        self._head_pose_tracker_menu.add_observer(
            "on_export_marker_tracker_3d_model_button_click",
            self._controller.export_marker_tracker_3d_model,
        )
        self._head_pose_tracker_menu.add_observer(
            "on_export_camera_traces_button_click",
            self._controller.export_camera_traces,
        )

        model_storage.add_observer("on_origin_marker_id_set", self._render_menu)

    def _on_init_ui(self):
        self._plugin.add_menu()
        self._plugin.menu.label = "Head Pose Tracker"
        self._visualization_3d_window.on_open_window()
        self._render_menu()

    def _on_deinit_ui(self):
        self._plugin.remove_menu()

    def _on_gl_display(self):
        self._display_2d_marker_detection()
        self._visualization_3d_window.on_display_3d_model()

    def _on_cleanup(self):
        self._visualization_3d_window.on_close_window()

    def _on_reset_button_click(self):
        self._controller.reset()
        self._render_menu()

    def _render_menu(self):
        self._plugin.menu.elements.clear()
        menu = self._head_pose_tracker_menu.create_menu()
        self._plugin.menu.extend(menu)

    def _display_2d_marker_detection(self):
        hat = np.array([[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]])
        for marker in self._controller_storage.marker_id_to_detections.values():
            hat_perspective = cv2.perspectiveTransform(
                hat, square_marker_detect.m_marker_to_screen(marker)
            )
            hat_perspective.shape = 6, 2

            cygl_utils.draw_polyline(
                hat_perspective,
                color=cygl_utils.RGBA(0.1, 1.0, 1.0, 0.2),
                line_type=gl.GL_POLYGON,
            )
