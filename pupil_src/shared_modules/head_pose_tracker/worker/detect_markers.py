"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import numpy as np

from apriltag.python import apriltag

detector_options = apriltag.DetectorOptions(families="tag36h11", nthreads=4)
detector = apriltag.Detector(detector_options)

_n_bins_x = 2
_n_bins_y = 2
_bins_x = np.linspace(0, 1, _n_bins_x + 1)[1:-1]
_bins_y = np.linspace(0, 1, _n_bins_y + 1)[1:-1]


def detect(frame):
    apriltag_detections = _detect(frame.gray)
    marker_detections = _get_marker_detections(
        apriltag_detections, [frame.width, frame.height]
    )
    return marker_detections


def _detect(frame_gray):
    return detector.detect(frame_gray)


def _get_marker_detections(apriltag_detections, frame_shape):
    return {
        apriltag_detection.tag_id: {
            "verts": apriltag_detection.corners[::-1].tolist(),
            "bin": _get_bin(apriltag_detection, frame_shape),
        }
        for apriltag_detection in apriltag_detections
        if apriltag_detection.hamming == 0
    }


def _get_bin(apriltag_detection, frame_shape):
    centroid = apriltag_detection.center / frame_shape
    bin_x = int(np.digitize(centroid[0], _bins_x))
    bin_y = int(np.digitize(centroid[1], _bins_y))
    return bin_x, bin_y


class marker_detection_callable:
    def __init__(self):
        pass

    def __call__(self, frame):
        detection = detect(frame)
        # print(detection)
        return detection
