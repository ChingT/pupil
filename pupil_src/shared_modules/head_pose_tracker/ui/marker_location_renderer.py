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


class MarkerLocationRenderer:
    """
    Renders marker locations in the world video.
    """

    def __init__(self, marker_location_storage, plugin, get_current_frame_index):
        self._marker_location_storage = marker_location_storage

        self.square_definition = np.array(
            ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32
        )
        self.hat_definition = np.array(
            [[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32
        )

        self._setup_glfont()

        self._get_current_frame_index = get_current_frame_index

        plugin.add_observer("gl_display", self.on_gl_display)

    def _setup_glfont(self):
        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", get_opensans_font_path())
        self.glfont.set_size(30)
        self.glfont.set_color_float((0.8, 0.2, 0.1, 0.8))
        self.glfont.set_align_string(v_align="center")

    def on_gl_display(self):
        self._render_marker_locations()

    def _render_marker_locations(self):
        current_index = self._get_current_frame_index()
        current_markers = self._marker_location_storage.get_or_none(current_index)
        if current_markers:
            self._render_2d_marker_boundary(current_markers.marker_detection)

    def _render_2d_marker_boundary(self, marker_detection):
        for marker_id, detection in marker_detection.items():
            perspective_matrix = cv2.getPerspectiveTransform(
                self.square_definition, np.array(detection["verts"], dtype=np.float32)
            )
            hat_points = cv2.perspectiveTransform(
                self.hat_definition, perspective_matrix
            )
            hat_points.shape = 6, 2

            color = (0.0, 1.0, 1.0, 0.2)
            self._draw_hat(hat_points, color)

    @staticmethod
    def _draw_hat(points, color):
        cygl_utils.draw_polyline(points, 1, cygl_utils.RGBA(*color), gl.GL_POLYGON)

    def _draw_marker_id(self, marker_detection):
        for (marker_id, detection) in marker_detection.items():
            point = np.max(detection["verts"], axis=0)
            self.glfont.draw_text(point[0], point[1], str(marker_id))
