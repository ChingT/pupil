"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


import aruco
import cv2
import numpy as np

from apriltag.python import apriltag


class ArucoDetectorCV2:
    def __init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.markers = {}
        self.color = (0.12, 0.46, 0.70)

    def detect(self, frame):
        corners, ids, _ = cv2.aruco.detectMarkers(frame.gray, self.aruco_dict)
        if corners:
            self.markers = {
                marker_id[0]: detection[0] for marker_id, detection in zip(ids, corners)
            }
        else:
            self.markers = {}


class ArucoDetectorPython:
    def __init__(self):
        self.detector = aruco.MarkerDetector()
        self.detector.setDetectionMode(aruco.DM_FAST)
        self.detector.setDictionary("ARUCO")  # "TAG36h11", "ARUCO_MIP_36h12"
        self.markers = {}
        self.color = (1.0, 0.49, 0.05)

    def detect(self, frame):
        self.aruco(frame)

    def aruco(self, frame):
        self.markers = {
            detection.id: np.array(detection)
            for detection in self.detector.detect(frame.bgr)
        }


class ApriltagDetector:
    def __init__(self):
        detector_options = apriltag.DetectorOptions(
            families="tag36h11",
            border=1,
            nthreads=1,
            quad_decimate=1.0,
            quad_blur=0.0,
            refine_edges=True,
            refine_decode=False,
            refine_pose=False,
            debug=False,
            quad_contours=True,
        )
        self._detector = apriltag.Detector(detector_options)
        self.markers = {}
        self.color = (0.17, 0.62, 0.17)

    def detect(self, frame):
        self.apriltag(frame)

    def apriltag(self, frame):
        apriltag_detections = self._detector.detect(frame.gray)
        self.markers = {
            detection.tag_id: np.array(detection.corners, dtype=np.float32)
            for detection in apriltag_detections
        }


class MarkerDetectorsController:
    def __init__(self, storage, markers_renderer, plugin):
        self._storage = storage
        self._markers_renderer = markers_renderer
        self._setup_detectors()

        plugin.add_observer("recent_events", self._on_recent_events)
        plugin.add_observer("gl_display", self._on_gl_display)

    def _setup_detectors(self):
        aruco_detector_cv2 = ArucoDetectorCV2()
        aruco_detector_python = ArucoDetectorPython()
        apriltag_detector = ApriltagDetector()

        self.detector_1 = aruco_detector_cv2
        self.detector_2 = aruco_detector_python
        self.detector_3 = apriltag_detector

    def _on_recent_events(self, events):
        try:
            frame = events["frame"]
        except KeyError:
            return

        if self._storage.show_aruco_detector_cv2:
            self.detector_1.detect(frame)
        if self._storage.show_aruco_detector_python:
            self.detector_2.detect(frame)
        if self._storage.show_apriltag_detector:
            self.detector_3.detect(frame)

    def _on_gl_display(self):
        if self._storage.show_aruco_detector_cv2:
            self._markers_renderer.render(
                self.detector_1.markers, self.detector_1.color
            )

        if self._storage.show_aruco_detector_python:
            self._markers_renderer.render(
                self.detector_2.markers, self.detector_2.color
            )

        if self._storage.show_apriltag_detector:
            self._markers_renderer.render(
                self.detector_3.markers, self.detector_3.color
            )
