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


class ApriltagMarkerDetector:
    def __init__(
        self,
        families="tag36h11",
        border=1,
        nthreads=4,
        quad_decimate=1.0,
        quad_blur=0.0,
        refine_edges=True,
        refine_decode=False,
        refine_pose=False,
        debug=False,
        quad_contours=True,
    ):
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

    def detect(self, image):
        apriltag_detections = self._detector.detect(image)

        img_size = image.shape[::-1]

        return {
            detection.tag_id: {
                "verts": detection.corners[::-1].tolist(),
                "centroid": normalize(detection.center, img_size, flip_y=True),
            }
            for detection in apriltag_detections
        }
