"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from aruco_tracker import ArucoDetectorCV2, ArucoDetectorPython, ApriltagDetector
from observable import Observable
from plugin import Plugin


class Aruco_Tracker(Plugin, Observable):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        aruco_detector_cv2 = ArucoDetectorCV2()
        aruco_detector_python = ArucoDetectorPython()
        apriltag_detector = ApriltagDetector()

        self.detector_1 = aruco_detector_cv2
        self.detector_2 = aruco_detector_python
        self.detector_3 = apriltag_detector

        self.add_observer("recent_events", self.detector_1.on_recent_events)
        self.add_observer("recent_events", self.detector_2.on_recent_events)
        self.add_observer("recent_events", self.detector_3.on_recent_events)

        self.add_observer("gl_display", self.detector_1.on_gl_display)
        self.add_observer("gl_display", self.detector_2.on_gl_display)
        self.add_observer("gl_display", self.detector_3.on_gl_display)
