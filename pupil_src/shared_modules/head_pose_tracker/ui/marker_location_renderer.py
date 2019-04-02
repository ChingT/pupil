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
    Renders 2d marker locations in the world video.
    """

    def __init__(
        self,
        marker_location_storage,
        markers_3d_model_storage,
        plugin,
        get_current_frame_index,
    ):
        self._get_current_frame_index = get_current_frame_index

        self._marker_locations = marker_location_storage.item
        self._markers_3d_model = markers_3d_model_storage.item

        self._square_definition = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
        )
        self._hat_definition = np.array(
            [[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32
        )

        self._setup_glfont()

        plugin.add_observer("gl_display", self._on_gl_display)

    def _setup_glfont(self):
        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", get_opensans_font_path())
        self.glfont.set_size(20)
        self.glfont.set_color_float((0.8, 0.2, 0.1, 0.8))

    def _on_gl_display(self):
        self._render()

    def _render(self):
        current_markers = self._get_current_markers()
        marker_id_optimized = self._get_marker_id_optimized()

        self._render_markers(current_markers, marker_id_optimized)

    def _get_current_markers(self):
        current_index = self._get_current_frame_index()
        try:
            return self._marker_locations.result[current_index]["marker_detection"]
        except KeyError:
            return {}

    def _get_marker_id_optimized(self):
        try:
            return self._markers_3d_model.result["marker_id_to_extrinsics"].keys()
        except TypeError:
            return []

    def _render_markers(self, current_markers, marker_id_optimized):
        for marker_id, detection in current_markers.items():
            marker_points = np.array(detection["verts"], dtype=np.float32)
            hat_points = self._calculate_hat_points(marker_points)
            if marker_id in marker_id_optimized:
                color = (1.0, 0.0, 0.0, 0.2)
            else:
                color = (0.0, 1.0, 1.0, 0.2)

            self._draw_hat(hat_points, color)

            if self._markers_3d_model.show_marker_id:
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
