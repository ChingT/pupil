"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import logging

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np

import gl_utils
import glfw

logger = logging.getLogger(__name__)


class Visualization3dWindow:
    def __init__(
        self,
        marker_location_storage,
        markers_3d_model_storage,
        camera_localizer_storage,
        camera_intrinsics,
        plugin,
        get_current_frame_index,
    ):
        self._marker_location_storage = marker_location_storage
        self._markers_3d_model_storage = markers_3d_model_storage
        self._camera_localizer_storage = camera_localizer_storage
        self._camera_intrinsics = camera_intrinsics
        self._plugin = plugin
        self._get_current_frame_index = get_current_frame_index

        self._input = {"down": False, "mouse": (0, 0)}

        self._init_trackball()

        self._window = None
        self._window_position = 0, 0
        self._window_size = 1280, 1335

        self.recent_camera_trace = collections.deque(maxlen=300)

        plugin.add_observer("init_ui", self._on_init_ui)
        plugin.add_observer("deinit_ui", self._on_deinit_ui)
        plugin.add_observer("gl_display", self._on_gl_display)

        glut.glutInit()

    def _init_trackball(self):
        self._trackball = gl_utils.trackball.Trackball()
        self._trackball.zoom_to(-150)

    def _on_init_ui(self):
        self.open()

    def _on_deinit_ui(self):
        self.close()

    def _on_cleanup(self):
        self.close()

    def _on_gl_display(self):
        self._display_3d_model(self._window)

    def open(self):
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

    def close(self):
        if not self._window:
            return

        self._window_position = glfw.glfwGetWindowPos(self._window)
        self._window_size = glfw.glfwGetWindowSize(self._window)
        glfw.glfwDestroyWindow(self._window)
        self._window = None
        logger.info("3d visualization window is closed")

    def _register_callbacks(self, window):
        glfw.glfwSetFramebufferSizeCallback(window, self._on_resize_window)
        glfw.glfwSetKeyCallback(window, self._on_window_key)
        glfw.glfwSetMouseButtonCallback(window, self._on_window_mouse_button)
        glfw.glfwSetCursorPosCallback(window, self._on_window_pos)
        glfw.glfwSetScrollCallback(window, self._on_scroll)
        glfw.glfwSetWindowCloseCallback(window, self._on_close_window)

    @staticmethod
    def _gl_state_settings(window):
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        gl_utils.basic_gl_setup()
        gl_utils.make_coord_system_norm_based()
        glfw.glfwSwapInterval(0)
        glfw.glfwMakeContextCurrent(active_window)

    def _on_resize_window(self, window, w, h):
        self._trackball.set_window_size(w, h)
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        gl_utils.adjust_gl_view(w, h)
        glfw.glfwMakeContextCurrent(active_window)

    def _on_window_key(self, window, key, scancode, action, mods):
        if action == glfw.GLFW_PRESS:
            if key == glfw.GLFW_KEY_RIGHT:
                self._trackball.pan_to(1, 0)
            elif key == glfw.GLFW_KEY_LEFT:
                self._trackball.pan_to(-1, 0)
            elif key == glfw.GLFW_KEY_DOWN:
                self._trackball.pan_to(0, 1)
            elif key == glfw.GLFW_KEY_UP:
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

    def _on_close_window(self, window):
        self.close()

    def _display_3d_model(self, window):
        if not window:
            return

        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        self._init_3d_window()
        self._trackball.push()

        self._render_centroid()
        markers_3d_model = self._markers_3d_model_storage.get_or_none()
        if markers_3d_model is not None:
            self._shift_rotate_center(markers_3d_model)
            self._render_coordinate_in_3d_window()
            self._render_markers(markers_3d_model)
            self._render_camera()

        self._trackball.pop()
        glfw.glfwSwapBuffers(window)
        glfw.glfwMakeContextCurrent(active_window)

    @staticmethod
    def _init_3d_window():
        gl.glClearColor(0.9, 0.9, 0.9, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearDepth(1.0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_DEPTH_TEST)

    @staticmethod
    def _render_centroid():
        gl.glLoadIdentity()
        gl.glPointSize(5)
        color = (0.2, 0.2, 0.2, 0.1)
        gl.glColor4f(*color)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(0, 0, 0)
        gl.glEnd()

    def _shift_rotate_center(self, markers_3d_model):
        camera_pose_matrix = np.eye(4, dtype=np.float32)
        camera_pose_matrix[0:3, 3] = -markers_3d_model.centroid
        gl.glLoadTransposeMatrixf(camera_pose_matrix)

    def _render_coordinate_in_3d_window(self, scale=1):
        color = (1, 0, 0, 1)
        self._render_line_in_3d_window((0, 0, 0), (scale, 0, 0), color)

        color = (0, 1, 0, 1)
        self._render_line_in_3d_window((0, 0, 0), (0, scale, 0), color)

        color = (0, 0, 1, 1)
        self._render_line_in_3d_window((0, 0, 0), (0, 0, scale), color)

    def _render_markers(self, markers_3d_model):
        self._render_3d_marker_boundary(markers_3d_model.result_vis)

        if markers_3d_model.show_marker_id:
            self._render_marker_id(markers_3d_model.result_vis)

    def _render_3d_marker_boundary(self, result_vis):
        current_index = self._get_current_frame_index()
        current_markers = self._marker_location_storage.get_or_none(current_index)

        if current_markers:
            current_markers_marker_detection = current_markers.marker_detection
        else:
            current_markers_marker_detection = {}

        for marker_id, points_3d in result_vis.items():
            if marker_id in current_markers_marker_detection:
                color = (1, 0, 0, 0.15)
            else:
                color = (1, 0, 0, 0.1)

            self._render_polygon_in_3d_window(points_3d, color)

    def _render_marker_id(self, result_vis):
        color = (1, 0, 0, 1)
        for (marker_id, points_3d) in result_vis.items():
            self._render_text_in_3d_window(str(marker_id), points_3d[0], color)

    def _render_camera(self):
        camera_localizer = self._camera_localizer_storage.get_or_none()
        if camera_localizer is None:
            return

        current_frame_index = self._get_current_frame_index()
        ts = self._plugin.g_pool.timestamps[current_frame_index]

        try:
            pose_datum = camera_localizer.pose_bisector.by_ts(ts)
        except ValueError:
            camera_trace = np.full((3,), np.nan)
            camera_pose_matrix = np.full((4, 4), np.nan)
        else:
            camera_trace = pose_datum["camera_trace"]
            camera_pose_matrix = pose_datum["camera_pose_matrix"]

        # recent_camera_trace is updated no matter show_camera_trace is on or not
        self.recent_camera_trace.append(camera_trace)

        if camera_localizer.show_camera_trace:
            self._render_camera_trace_in_3d_window(self.recent_camera_trace)

        self._render_camera_frustum(camera_pose_matrix)

    def _render_camera_trace_in_3d_window(self, recent_camera_traces):
        color = (0.2, 0.2, 0.2, 0.1)
        self._render_strip_in_3d_window(recent_camera_traces, color)

    def _render_camera_frustum(self, camera_pose_matrix):
        if camera_pose_matrix is not None:
            gl.glMultTransposeMatrixf(camera_pose_matrix)
            self._render_coordinate_in_3d_window()
            self._render_frustum_in_3d_window(
                self._camera_intrinsics.resolution, self._camera_intrinsics.K
            )

    def _render_frustum_in_3d_window(self, img_size, camera_intrinsics, scale=1000):
        x = img_size[0] / scale
        y = img_size[1] / scale
        z = (camera_intrinsics[0, 0] + camera_intrinsics[1, 1]) / scale

        vertices = []
        vertices += [[0, 0, 0], [x, y, z], [x, -y, z]]
        vertices += [[0, 0, 0], [x, y, z], [-x, y, z]]
        vertices += [[0, 0, 0], [-x, -y, z], [x, -y, z]]
        vertices += [[0, 0, 0], [-x, -y, z], [-x, y, z]]

        color = (0.05, 0.05, 0.05, 0.1)
        self._render_polygon_in_3d_window(vertices, color)

    @staticmethod
    def _render_line_in_3d_window(start_point, end_point, color):
        gl.glColor4f(*color)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(*start_point)
        gl.glVertex3f(*end_point)
        gl.glEnd()

    @staticmethod
    def _render_lines_in_3d_window(vertices, color):
        gl.glColor4f(*color)
        gl.glBegin(gl.GL_LINES)
        for vertex in vertices:
            gl.glVertex3f(*vertex)
        gl.glEnd()

    @staticmethod
    def _render_polygon_in_3d_window(vertices, color):
        r, g, b, _ = color
        gl.glColor4f(r, g, b, 0.5)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glBegin(gl.GL_POLYGON)
        for vertex in vertices:
            gl.glVertex3f(*vertex)
        gl.glEnd()

        gl.glColor4f(*color)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glBegin(gl.GL_POLYGON)
        for vertex in vertices:
            gl.glVertex3f(*vertex)
        gl.glEnd()

    @staticmethod
    def _render_strip_in_3d_window(vertices, color):
        gl.glColor4f(*color)
        gl.glBegin(gl.GL_LINE_STRIP)
        for vertex in vertices:
            gl.glVertex3f(*vertex)
        gl.glEnd()

    @staticmethod
    def _render_text_in_3d_window(characters, position, color):
        gl.glColor4f(*color)
        gl.glRasterPos3f(*position)
        for character in characters:
            glut.glutBitmapCharacter(glut.GLUT_BITMAP_8_BY_13, ord(character))
