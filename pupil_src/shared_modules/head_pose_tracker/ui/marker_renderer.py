"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import OpenGL.GL as gl
import cv2
import numpy as np
import pyglui.cygl.utils as cygl_utils
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path


class MarkerRenderer:
    """
    Renders markers in the world video.
    """

    def __init__(self, controller_storage, model_storage, plugin):
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._plugin = plugin

        self.square_definition = np.array(
            ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32
        )
        self.hat_definition = np.array(
            [[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32
        )

        self._setup_glfont()

        plugin.add_observer("gl_display", self.on_gl_display)

    def _setup_glfont(self):
        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", get_opensans_font_path())
        self.glfont.set_size(30)
        self.glfont.set_color_float((0.8, 0.2, 0.1, 0.8))
        self.glfont.set_align_string(v_align="center")

    def on_gl_display(self):
        self._draw_marker_boundary()
        if self._plugin.visualization_3d_window.show_marker_id:
            self._draw_marker_id()

    def _draw_marker_boundary(self):
        for (
            marker_id,
            detection,
        ) in self._controller_storage.marker_id_to_detections.items():
            perspective_matrix = cv2.getPerspectiveTransform(
                self.square_definition, detection["verts"]
            )
            hat_points = cv2.perspectiveTransform(
                self.hat_definition, perspective_matrix
            )
            hat_points.shape = 6, 2

            if marker_id == self._model_storage.origin_marker_id:
                color = (0.8, 0.2, 0.1, 0.5)
            elif marker_id in self._model_storage.marker_id_to_extrinsics_opt:
                color = (0.8, 0.2, 0.1, 0.2)
            # TODO: debug only; to be removed
            elif marker_id in self._model_storage.marker_id_to_points_3d_init:
                color = (0.0, 0.0, 1.0, 0.2)
            else:
                color = (0.0, 1.0, 1.0, 0.2)

            self._draw_hat(hat_points, color)

    @staticmethod
    def _draw_hat(points, color):
        cygl_utils.draw_polyline(points, 1, cygl_utils.RGBA(*color), gl.GL_POLYGON)

    def _draw_marker_id(self):
        for (
            marker_id,
            detection,
        ) in self._controller_storage.marker_id_to_detections.items():
            point = np.max(detection["verts"], axis=0)
            self.glfont.draw_text(point[0], point[1], str(marker_id))
