"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

import OpenGL.GL as gl
import cv2
import numpy as np
import pyglui.cygl.utils as cygl_utils
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

logger = logging.getLogger(__name__)


class MarkersRenderer:
    def __init__(self):
        self._square_definition = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
        )
        self._hat_definition = np.array(
            [[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32
        )

        self._setup_glfont()

    def _setup_glfont(self):
        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", get_opensans_font_path())
        self.glfont.set_size(20)

    def render(self, markers, color):
        r, g, b = color
        self.glfont.set_color_float((r, g, b, 1))
        for marker_id, marker_points in markers.items():
            hat_points = self._calculate_hat_points(marker_points)

            self._draw_hat(hat_points, (r, g, b, 0.3))
            self._draw_marker_id(marker_points, marker_id)

    def _calculate_hat_points(self, marker_points):
        perspective_matrix = cv2.getPerspectiveTransform(
            self._square_definition, marker_points
        )
        hat_points = cv2.perspectiveTransform(self._hat_definition, perspective_matrix)
        hat_points.shape = 6, 2
        return hat_points

    @staticmethod
    def _draw_hat(points, color):
        cygl_utils.draw_polyline(points, 1, cygl_utils.RGBA(*color), gl.GL_POLYGON)

    def _draw_marker_id(self, marker_points, marker_id):
        point = np.max(marker_points, axis=0)
        self.glfont.draw_text(point[0], point[1], str(marker_id))
