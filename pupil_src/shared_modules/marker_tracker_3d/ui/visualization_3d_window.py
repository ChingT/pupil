import logging

import OpenGL.GL as gl

import gl_utils
import glfw

logger = logging.getLogger(__name__)


class Visualization3dWindow:
    def __init__(self, intrinsics, controller_storage, model_storage):
        self._intrinsics = intrinsics

        self._max_camera_traces_len = 200
        self._input = {"down": False, "mouse": (0, 0)}

        self._init_trackball()

        self._window = None
        self._window_position = 0, 0
        self._window_size = 1280, 1335

        self._controller_storage = controller_storage
        self._model_storage = model_storage

    def _init_trackball(self):
        self._trackball = gl_utils.trackball.Trackball()
        self._trackball.zoom_to(-100)

    def on_open_window(self, window=None):
        self._open_window()

    def on_close_window(self, window=None):
        self._close_window()

    def on_display_3d_model(self):
        self._display_3d_model(self._window)

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
        for (
            marker_id,
            points_3d,
        ) in self._model_storage.marker_id_to_points_3d_opt.items():
            if marker_id in self._controller_storage.marker_id_to_detections:
                color = (1, 0, 0, 0.8)
            else:
                color = (1, 0.4, 0, 0.6)

            gl.glColor4f(*color)
            self._draw_polygon_in_3d_window(points_3d)

    def _draw_camera_trace_in_3d_window(self):
        trace = self._controller_storage.all_camera_traces[
            -self._max_camera_traces_len :
        ]
        gl.glColor4f(0, 0, 0.8, 0.2)
        for i in range(len(trace) - 1):
            self._draw_line_in_3d_window(trace[i], trace[i + 1])

    def _draw_camera_in_3d_window(self):
        try:
            camera_pose_matrix_flatten = (
                self._controller_storage.camera_pose_matrix.T.flatten()
            )
        except AttributeError:
            pass
        else:
            self._draw_frustum_in_3d_window(
                camera_pose_matrix_flatten,
                self._intrinsics.resolution,
                self._intrinsics.K,
            )
            self._draw_coordinate_in_3d_window()

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
        glfw.glfwSetFramebufferSizeCallback(window, self._on_resize_window)
        glfw.glfwSetKeyCallback(window, self._on_window_key)
        glfw.glfwSetMouseButtonCallback(window, self._on_window_mouse_button)
        glfw.glfwSetCursorPosCallback(window, self._on_window_pos)
        glfw.glfwSetScrollCallback(window, self._on_scroll)
        glfw.glfwSetWindowCloseCallback(window, self.on_close_window)

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
