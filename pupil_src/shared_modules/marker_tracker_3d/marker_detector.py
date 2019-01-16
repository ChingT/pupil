import logging

import numpy as np

import square_marker_detect

logger = logging.getLogger(__name__)


class MarkerDetector:
    def __init__(self, min_marker_perimeter):
        self.min_marker_perimeter = min_marker_perimeter

    def detect(self, frame):
        # not use detect_markers_robust to avoid cv2.calcOpticalFlowPyrLK for
        # performance reasons
        try:
            marker_list = square_marker_detect.detect_markers(
                frame.gray,
                grid_size=5,
                aperture=13,
                min_marker_perimeter=self.min_marker_perimeter,
            )
        except AttributeError:
            return {}
        else:
            marker_list = self._filter_markers(marker_list)
            marker_detections = {
                m["id"]: {
                    "verts": m["verts"],
                    "centroid": np.array(m["centroid"]) / [frame.width, frame.height],
                }
                for m in marker_list
            }
            return marker_detections

    def _filter_markers(self, marker_list):
        marker_list = [m for m in marker_list if m["id_confidence"] > 0.9]

        markers_id_all = set([m["id"] for m in marker_list])
        for marker_id in markers_id_all:
            marker_list = self._remove_duplicate(marker_id, marker_list)

        return marker_list

    def _remove_duplicate(self, marker_id, marker_list):
        markers_with_same_id = [m for m in marker_list if m["id"] == marker_id]
        if len(markers_with_same_id) < 2:
            pass
        elif len(markers_with_same_id) == 2 and self._check_two_duplicate_detections(
            markers_with_same_id
        ):
            # If there are two markers with the same id very close to each other,
            # we assume there are duplicate marker detections
            # and will discard the smaller marker
            marker_small = min(markers_with_same_id, key=lambda x: x["perimeter"])
            marker_list = [
                m
                for m in marker_list
                if not (
                    m["id"] == marker_id and m["centroid"] == marker_small["centroid"]
                )
            ]
        else:
            # If there are two markers with the same id not close to each other
            # or more than two markers with the same id are detected,
            # we assume multiple markers with the same id are displayed in the frame
            # and will remove duplicate markers.
            marker_list = [m for m in marker_list if m["id"] != marker_id]
            logger.warning(
                "WARNING! Multiple markers with the same id {} found!".format(marker_id)
            )
        return marker_list

    @staticmethod
    def _check_two_duplicate_detections(markers_with_same_id):
        dist = np.linalg.norm(
            np.array(markers_with_same_id[0]["centroid"])
            - np.array(markers_with_same_id[1]["centroid"])
        )
        return bool(dist < 5)
