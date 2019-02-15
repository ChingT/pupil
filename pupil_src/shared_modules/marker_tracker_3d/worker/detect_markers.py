from apriltag.python import apriltag

detector_options = apriltag.DetectorOptions(families="tag36h11", nthreads=1)
detector = apriltag.Detector(detector_options)


def detect(frame):
    apriltag_detections = _detect(frame.gray)

    marker_detections = {
        apriltag_detection.tag_id: {
            "verts": apriltag_detection.corners[::-1],
            "centroid": apriltag_detection.center / [frame.width, frame.height],
        }
        for apriltag_detection in apriltag_detections
    }
    return marker_detections


def _detect(frame_gray):
    return detector.detect(frame_gray)
