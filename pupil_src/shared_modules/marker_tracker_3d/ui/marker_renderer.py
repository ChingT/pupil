import OpenGL.GL as gl
import cv2
import numpy as np
import pyglui.cygl.utils as cygl_utils

import square_marker_detect


class MarkerRenderer:
    """
    Renders markers in the world video.
    """

    def __init__(self, controller_storage, plugin):
        self._controller_storage = controller_storage

        self.hat = np.array([[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]])

        plugin.add_observer("gl_display", self.on_gl_display)

    def on_gl_display(self):
        self._draw_2d_marker_detection()

    def _draw_2d_marker_detection(self):
        for marker in self._controller_storage.marker_id_to_detections.values():
            hat_perspective = cv2.perspectiveTransform(
                self.hat, square_marker_detect.m_marker_to_screen(marker)
            )
            hat_perspective.shape = 6, 2
            self._draw_hat(hat_perspective)

    @staticmethod
    def _draw_hat(points):
        cygl_utils.draw_polyline(
            points, color=cygl_utils.RGBA(0.1, 1.0, 1.0, 0.2), line_type=gl.GL_POLYGON
        )
