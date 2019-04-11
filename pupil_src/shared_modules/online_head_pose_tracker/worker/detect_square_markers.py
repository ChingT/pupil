"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from apriltag.python import apriltag
from methods import normalize

apriltag_detector = apriltag.Detector()


def detect(frame):
    image = frame.gray
    apriltag_detections = apriltag_detector.detect(image)
    img_size = image.shape[::-1]
    return [
        {
            "id": detection.tag_id,
            "verts": detection.corners[::-1].tolist(),
            "centroid": normalize(detection.center, img_size, flip_y=True),
            "timestamp": frame.timestamp,
        }
        for detection in apriltag_detections
    ]
