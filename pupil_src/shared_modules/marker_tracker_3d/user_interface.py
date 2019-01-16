import logging

import OpenGL.GL as gl
import cv2
import numpy as np
from pyglui import ui as ui
from pyglui.cygl import utils as cygl_utils

import gl_utils
import glfw
import square_marker_detect

logger = logging.getLogger(__name__)


class UserInterface:
    def __init__(self, marker_tracker_3d, intrinsics):
        self._marker_tracker_3d = marker_tracker_3d
        self._intrinsics = intrinsics

        self._marker_tracker_3d.menu = None

        self._max_camera_traces_len = 200
        self._input = {"down": False, "mouse": (0, 0)}

        self._init_trackball()

        self._open_3d_window = True
        self._window = None
        self._window_position = 0, 0
        self._window_size = 1280, 1335
        self._open_close_window(self._open_3d_window)

        self._register_observers()

        self.model_state = (
            self._marker_tracker_3d.controller.model_optimizer.model_state
        )
        self.controller_storage = self._marker_tracker_3d.controller.storage

    def _init_trackball(self):
        self._trackball = gl_utils.trackball.Trackball()
        self._trackball.zoom_to(-100)

    def _register_observers(self):
        self._marker_tracker_3d.add_observer("init_ui", self._on_init_ui)
        self._marker_tracker_3d.add_observer("deinit_ui", self._on_deinit_ui)
        self._marker_tracker_3d.add_observer("gl_display", self._on_gl_display)
        self._marker_tracker_3d.controller.model_optimizer.visibility_graphs.add_observer(
            "set_up_origin_marker", self._on_update_menu
        )
        self._marker_tracker_3d.add_observer("cleanup", self._on_close_window)

    def _on_init_ui(self):
        self._marker_tracker_3d.add_menu()
        self._marker_tracker_3d.menu.label = "Head Pose Tracker"
        self._update_menu()

    def _on_deinit_ui(self):
        self._marker_tracker_3d.remove_menu()

    def _on_gl_display(self):
        self._display_2d_marker_detection()
        self._display_3d_model(self._window)

    def _on_update_menu(self):
        self._update_menu()

    def _on_close_window(self, window=None):
        self._open_close_window(open_3d_window=False)

    def _display_2d_marker_detection(self):
        hat = np.array([[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]])
        for marker in self.controller_storage.marker_id_to_detections.values():
            hat_perspective = cv2.perspectiveTransform(
                hat, square_marker_detect.m_marker_to_screen(marker)
            )
            hat_perspective.shape = 6, 2

            cygl_utils.draw_polyline(
                hat_perspective,
                color=cygl_utils.RGBA(0.1, 1.0, 1.0, 0.2),
                line_type=gl.GL_POLYGON,
            )

    def _display_3d_model(self, window):
        if not window:
            return

        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)

        self._init_3d_window()
        self._trackball.push()
        self._draw_coordinate_in_3d_window()
        self._draw_markers_in_3d_window()
        self._draw_camera_trace_in_3d_window()
        self._draw_camera_in_3d_window()
        self._trackball.pop()

        glfw.glfwSwapBuffers(window)
        glfw.glfwMakeContextCurrent(active_window)

    @staticmethod
    def _init_3d_window():
        gl.glClearColor(0.8, 0.8, 0.8, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearDepth(1.0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _draw_coordinate_in_3d_window(self):
        gl.glColor4f(1, 0, 0, 0.5)
        self._draw_line_in_3d_window((0, 0, 0), (1, 0, 0))

        gl.glColor4f(0, 1, 0, 0.5)
        self._draw_line_in_3d_window((0, 0, 0), (0, 1, 0))

        gl.glColor4f(0, 0, 1, 0.5)
        self._draw_line_in_3d_window((0, 0, 0), (0, 0, 1))

    def _draw_markers_in_3d_window(self):
        for idx, vertices in self.model_state.marker_points_3d_opt.items():
            if idx in self.controller_storage.marker_id_to_detections:
                color = (1, 0, 0, 0.8)
            else:
                color = (1, 0.4, 0, 0.6)

            gl.glColor4f(*color)
            self._draw_polygon_in_3d_window(vertices)

    def _draw_camera_trace_in_3d_window(self):
        trace = self.controller_storage.all_camera_traces[
            -self._max_camera_traces_len :
        ]
        gl.glColor4f(0, 0, 0.8, 0.2)
        for i in range(len(trace) - 1):
            self._draw_line_in_3d_window(trace[i], trace[i + 1])

    def _draw_camera_in_3d_window(self):
        try:
            camera_pose_matrix_flatten = (
                self.controller_storage.current_camera_pose_matrix.T.flatten()
            )
        except AttributeError:
            pass
        else:
            self._draw_coordinate_in_3d_window()
            self._draw_frustum_in_3d_window(
                camera_pose_matrix_flatten,
                self._intrinsics.resolution,
                self._intrinsics.K,
            )

    def _draw_frustum_in_3d_window(
        self, matrix_flatten, img_size, camera_intrinsics, scale=1000
    ):
        x = img_size[0] / scale
        y = img_size[1] / scale
        z = (camera_intrinsics[0, 0] + camera_intrinsics[1, 1]) / scale

        vertices = []
        vertices += [[0, 0, 0], [x, y, z], [x, -y, z]]
        vertices += [[0, 0, 0], [x, y, z], [-x, y, z]]
        vertices += [[0, 0, 0], [-x, -y, z], [x, -y, z]]
        vertices += [[0, 0, 0], [-x, -y, z], [-x, y, z]]

        gl.glColor4f(0, 0, 0.6, 0.8)
        gl.glMultMatrixf(matrix_flatten)
        self._draw_polygon_in_3d_window(vertices)

    @staticmethod
    def _draw_line_in_3d_window(start_point, end_point):
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(*start_point)
        gl.glVertex3f(*end_point)
        gl.glEnd()

    @staticmethod
    def _draw_polygon_in_3d_window(vertices):
        gl.glBegin(gl.GL_LINE_LOOP)
        for vertex in vertices:
            gl.glVertex3f(*vertex)
        gl.glEnd()

    def _resize_window(self, window, w, h):
        self._trackball.set_window_size(w, h)
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        gl_utils.adjust_gl_view(w, h)
        glfw.glfwMakeContextCurrent(active_window)

    def _on_resize(self, window, w, h):
        self._resize_window(window, w, h)

    def _on_window_key(self, window, key, scancode, action, mods):
        if action == glfw.GLFW_PRESS:
            if key == glfw.GLFW_KEY_RIGHT:
                self._trackball.pan_to(1, 0)
            if key == glfw.GLFW_KEY_LEFT:
                self._trackball.pan_to(-1, 0)
            if key == glfw.GLFW_KEY_DOWN:
                self._trackball.pan_to(0, 1)
            if key == glfw.GLFW_KEY_UP:
                self._trackball.pan_to(0, -1)

    def _on_window_mouse_button(self, window, button, action, mods):
        if action == glfw.GLFW_PRESS:
            self._input["down"] = True
            self._input["mouse"] = glfw.glfwGetCursorPos(window)
        elif action == glfw.GLFW_RELEASE:
            self._input["down"] = False

    def _on_window_pos(self, window, x, y):
        if self._input["down"]:
            old_x, old_y = self._input["mouse"]
            self._trackball.drag_to(x - old_x, y - old_y)
            self._input["mouse"] = x, y

    def _on_scroll(self, window, x, y):
        self._trackball.zoom_to(y)

    def _update_menu(self):
        self._marker_tracker_3d.menu.elements[:] = []
        # self._marker_tracker_3d.menu.append()
        self._marker_tracker_3d.menu.extend(self._render_menu())

    def _render_menu(self):
        menu = [
            self._create_intro_text(),
            self._create_origin_marker_text(),
            self._create_min_marker_perimeter_slider(),
            self._create_open_3d_window_switch(),
            self._create_adding_marker_detections_switch(),
            self._create_reset_button(),
            self._create_export_model_button(),
            self._create_export_camera_traces_button(),
        ]
        return menu

    def _create_intro_text(self):
        return ui.Info_Text(
            "This plugin outputs current camera pose in relation to the printed "
            "markers in the scene"
        )

    def _create_origin_marker_text(self):
        text = self._get_text_for_origin_marker()
        return ui.Info_Text(text)

    def _create_min_marker_perimeter_slider(self):
        return ui.Slider(
            "min_marker_perimeter",
            self._marker_tracker_3d.controller.marker_detector,
            step=1,
            min=30,
            max=100,
            label="Perimeter of markers",
        )

    def _create_open_3d_window_switch(self):
        return ui.Switch(
            "_open_3d_window",
            self,
            label="3d visualization window",
            setter=self._open_close_window,
        )

    def _create_adding_marker_detections_switch(self):
        return ui.Switch(
            "adding_marker_detections",
            self._marker_tracker_3d.controller.model_optimizer.visibility_graphs,
            label="Adding observations",
        )

    def _create_reset_button(self):
        return ui.Button(label="reset", function=self._on_reset_button_click)

    def _create_export_model_button(self):
        return ui.Button(
            label="export marker tracker 3d model",
            function=self._on_export_marker_tracker_3d_model_button_click,
        )

    def _create_export_camera_traces_button(self):
        return ui.Button(
            label="export camera traces",
            function=self._on_export_camera_traces_button_click,
        )

    def _get_text_for_origin_marker(self):
        try:
            _origin_marker_id = self.model_state.marker_ids[0]
        except IndexError:
            text = "The coordinate system has not yet been built up"
        else:
            text = (
                "The marker with id {} is defined as the origin of the coordinate "
                "system".format(_origin_marker_id)
            )

        logger.info(text)
        return text

    def _on_reset_button_click(self):
        self._marker_tracker_3d.controller.reset()
        self._update_menu()

    def _on_export_marker_tracker_3d_model_button_click(self):
        self._marker_tracker_3d.controller.export_marker_tracker_3d_model()

    def _on_export_camera_traces_button_click(self):
        self._marker_tracker_3d.controller.export_camera_traces()

    def _open_close_window(self, open_3d_window):
        self._open_3d_window = open_3d_window
        if self._open_3d_window:
            self._open_window()
        else:
            self._close_window()

    def _open_window(self):
        if self._window:
            return

        self._window = glfw.glfwCreateWindow(
            *self._window_size,
            title="Head Pose Tracker",
            share=glfw.glfwGetCurrentContext()
        )
        glfw.glfwSetWindowPos(self._window, *self._window_position)
        self._gl_state_settings(self._window)
        self._register_callbacks(self._window)

        logger.info("3d visualization window is opened")

    def _close_window(self):
        if not self._window:
            return

        self._window_position = glfw.glfwGetWindowPos(self._window)
        self._window_size = glfw.glfwGetWindowSize(self._window)
        glfw.glfwDestroyWindow(self._window)
        self._window = None
        logger.info("3d visualization window is closed")

    def _register_callbacks(self, window):
        glfw.glfwSetFramebufferSizeCallback(window, self._on_resize)
        glfw.glfwSetKeyCallback(window, self._on_window_key)
        glfw.glfwSetWindowCloseCallback(window, self._on_close_window)
        glfw.glfwSetMouseButtonCallback(window, self._on_window_mouse_button)
        glfw.glfwSetCursorPosCallback(window, self._on_window_pos)
        glfw.glfwSetScrollCallback(window, self._on_scroll)

    @staticmethod
    def _gl_state_settings(window):
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        gl_utils.basic_gl_setup()
        gl_utils.make_coord_system_norm_based()
        glfw.glfwSwapInterval(0)
        glfw.glfwMakeContextCurrent(active_window)
