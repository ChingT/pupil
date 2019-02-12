from apriltag.python import apriltag


detector = apriltag.Detector()


def detect(frame):
    apriltag_detections = detector.detect(frame.gray)
    marker_detections = {
        apriltag_detection.tag_id: {
            "verts": apriltag_detection.corners[::-1],
            "centroid": apriltag_detection.center / [frame.width, frame.height],
        }
        for apriltag_detection in apriltag_detections
    }
    return marker_detections
