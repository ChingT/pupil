import OpenGL.GL as gl
import cv2
import numpy as np
import pyglui.cygl.utils as cygl_utils

import square_marker_detect


class MarkerRenderer:
    """
    Renders markers in the world video.
    """

    def __init__(self, controller_storage, model_storage, plugin):
        self._plugin = plugin
        self._controller_storage = controller_storage
        self._model_storage = model_storage

        self.hat = np.array([[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]])

        plugin.add_observer("gl_display", self.on_gl_display)

    def on_gl_display(self):
        self._draw_2d_marker_detection()

    def _draw_2d_marker_detection(self):
        for (
            marker_id,
            points,
        ) in self._controller_storage.marker_id_to_detections.items():
            hat_perspective = cv2.perspectiveTransform(
                self.hat, square_marker_detect.m_marker_to_screen(points)
            )
            hat_perspective.shape = 6, 2
            if marker_id == self._model_storage.origin_marker_id:
                color = (0.8, 0.2, 0.1, 0.5)
            elif marker_id in self._model_storage.marker_id_to_extrinsics_opt:
                color = (0.8, 0.2, 0.1, 0.2)
            # TODO: debug only; to be removed
            elif (
                self._plugin.head_pose_tracker_menu.show_markers_init
                and marker_id in self._model_storage.marker_id_to_points_3d_init
            ):
                color = (0.0, 0.0, 1.0, 0.2)
            else:
                color = (0.0, 1.0, 1.0, 0.2)
            self._draw_hat(hat_perspective, color)

    @staticmethod
    def _draw_hat(points, color):
        cygl_utils.draw_polyline(
            points, color=cygl_utils.RGBA(*color), line_type=gl.GL_POLYGON
        )
