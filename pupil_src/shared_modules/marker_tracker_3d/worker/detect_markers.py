import numpy as np

from apriltag.python import apriltag

detector_options = apriltag.DetectorOptions(families="tag36h11", nthreads=4)
detector = apriltag.Detector(detector_options)


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
            "verts": apriltag_detection.corners[::-1].astype(np.float32),
            "centroid": (apriltag_detection.center / frame_shape).astype(np.float32),
        }
        for apriltag_detection in apriltag_detections
        if apriltag_detection.hamming == 0
    }
