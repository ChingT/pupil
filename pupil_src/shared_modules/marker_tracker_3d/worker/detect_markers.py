from apriltag.python import apriltag


detector = apriltag.Detector()


def detect(frame):
    detections = detector.detect(frame.gray)

    marker_detections = {
        detection.tag_id: {
            "verts": detection.corners[::-1],
            "centroid": detection.center / [frame.width, frame.height],
        }
        for detection in detections
    }
    return marker_detections
