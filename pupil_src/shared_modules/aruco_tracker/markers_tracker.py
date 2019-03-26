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
from aruco_tracker import MarkersRenderer, timer


class ArucoDetectorCV2(MarkersRenderer):
    def __init__(self):
        super().__init__()

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    def _detect(self, frame):
        corners, ids, _ = cv2.aruco.detectMarkers(frame.gray, self.aruco_dict)
        if corners:
            self.markers = {
                marker_id[0]: detection[0] for marker_id, detection in zip(ids, corners)
            }
        else:
            self.markers = {}


class ArucoDetectorPython(MarkersRenderer):
    def __init__(self):
        super().__init__()

        self.detector = aruco.MarkerDetector()
        self.detector.setDetectionMode(aruco.DM_FAST)
        self.detector.setDictionary("ARUCO_MIP_36h12")  # "TAG36h11", "ARUCO_MIP_36h12"

    def _detect(self, frame):
        self.aruco(frame)

    @timer
    def aruco(self, frame):
        self.markers = {
            detection.id: np.array(detection)
            for detection in self.detector.detect(frame.bgr)
        }


class ApriltagDetector(MarkersRenderer):
    def __init__(
        self,
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
    ):
        super().__init__()

        detector_options = apriltag.DetectorOptions(
            families,
            border,
            nthreads,
            quad_decimate,
            quad_blur,
            refine_edges,
            refine_decode,
            refine_pose,
            debug,
            quad_contours,
        )
        self._detector = apriltag.Detector(detector_options)

    def _detect(self, frame):
        self.apriltag(frame)

    @timer
    def apriltag(self, frame):
        apriltag_detections = self._detector.detect(frame.gray)
        self.markers = {
            detection.tag_id: np.array(detection.corners, dtype=np.float32)
            for detection in apriltag_detections
        }