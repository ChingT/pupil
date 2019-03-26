"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import functools
import logging
import time

import OpenGL.GL as gl
import cv2
import numpy as np
import pyglui.cygl.utils as cygl_utils
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

logger = logging.getLogger(__name__)


class MarkersRenderer(abc.ABC):
    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self):
        self._square_definition = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
        )
        self._hat_definition = np.array(
            [[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32
        )

        self._setup_glfont()

        self.markers = {}

    def _setup_glfont(self):
        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", get_opensans_font_path())
        self.glfont.set_size(20)
        self.glfont.set_color_float((0.8, 0.2, 0.1, 0.8))

    def on_recent_events(self, events):
        try:
            frame = events["frame"]
        except KeyError:
            return
        self._detect(frame)

    @abc.abstractmethod
    def _detect(self, frame):
        pass

    def on_gl_display(self):
        for marker_id, marker_points in self.markers.items():
            hat_points = self._calculate_hat_points(marker_points)
            color = (1.0, 0.0, 0.0, 0.2)

            self._draw_hat(hat_points, color)
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


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        t1 = time.perf_counter()
        value = func(*args, **kwargs)
        t2 = time.perf_counter()
        run_time = t2 - t1
        if run_time > 1e-3:
            logger.info("{0} took {1:.2f} ms".format(func.__name__, run_time * 1e3))
        else:
            logger.info("{0} took {1:.2f} µs".format(func.__name__, run_time * 1e6))

        return value

    return wrapper_timer
