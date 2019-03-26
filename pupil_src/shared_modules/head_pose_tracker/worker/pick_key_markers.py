"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections

import numpy as np

KeyMarker = collections.namedtuple(
    "KeyMarker", ["frame_id", "marker_id", "verts", "bin"]
)


class PickKeyMarkers:
    def __init__(
        self,
        optimization_storage,
        select_key_markers_interval=2,
        min_n_markers_per_frame=2,
        max_n_same_markers_per_bin=1,
        n_bins_x=2,
        n_bins_y=2,
    ):
        assert select_key_markers_interval >= 1
        assert min_n_markers_per_frame >= 2
        assert max_n_same_markers_per_bin >= 1

        self._optimization_storage = optimization_storage
        self._select_key_markers_interval = select_key_markers_interval
        self._min_n_markers_per_frame = min_n_markers_per_frame
        self._max_n_same_markers_per_bin = max_n_same_markers_per_bin

        self._bins_x = np.linspace(0, 1, n_bins_x + 1)[1:-1]
        self._bins_y = np.linspace(0, 1, n_bins_y + 1)[1:-1]

        self._n_frames_passed = -1

    def run(self, marker_id_to_detections, timestamp):
        if self._decide_key_markers(marker_id_to_detections):
            self._save_key_markers(marker_id_to_detections, timestamp)

    def _decide_key_markers(self, marker_id_to_detections):
        self._n_frames_passed += 1
        if self._n_frames_passed >= self._select_key_markers_interval:
            self._n_frames_passed = -1

            if len(
                marker_id_to_detections
            ) >= self._min_n_markers_per_frame and self._check_bins_availability(
                marker_id_to_detections
            ):
                return True

        return False

    def _check_bins_availability(self, marker_id_to_detections):
        for marker_id, detection in marker_id_to_detections.items():
            n_same_markers_in_bin = len(
                [
                    marker
                    for marker in self._optimization_storage.all_key_markers
                    if marker.marker_id == marker_id
                    and marker.bin == self._get_bin(detection)
                ]
            )
            if n_same_markers_in_bin < self._max_n_same_markers_per_bin:
                return True

        return False

    def _save_key_markers(self, marker_id_to_detections, frame_id):
        key_markers = [
            KeyMarker(frame_id, marker_id, detection["verts"], self._get_bin(detection))
            for marker_id, detection in marker_id_to_detections.items()
        ]
        self._optimization_storage.all_key_markers += key_markers

    def _get_bin(self, detection):
        centroid = detection["centroid"]
        bin_x = int(np.digitize(centroid[0], self._bins_x))
        bin_y = int(np.digitize(centroid[1], self._bins_y))
        return bin_x, bin_y
