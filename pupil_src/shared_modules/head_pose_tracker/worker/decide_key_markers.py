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
import itertools as it

import numpy as np

KeyMarker = collections.namedtuple(
    "KeyMarker", ["frame_id", "marker_id", "verts", "bin"]
)


class DecideKeyMarkers:
    def __init__(
        self,
        controller_storage,
        select_key_markers_interval=3,
        min_n_markers_per_frame=2,
        max_n_same_markers_per_bin=1,
    ):
        assert select_key_markers_interval >= 1
        assert min_n_markers_per_frame >= 2
        assert max_n_same_markers_per_bin >= 1

        self._controller_storage = controller_storage
        self._select_key_markers_interval = select_key_markers_interval
        self._min_n_markers_per_frame = min_n_markers_per_frame
        self._max_n_same_markers_per_bin = max_n_same_markers_per_bin

        self._n_bins_x = 2
        self._n_bins_y = 2
        self._bins_x = np.linspace(0, 1, self._n_bins_x + 1)[1:-1]
        self._bins_y = np.linspace(0, 1, self._n_bins_y + 1)[1:-1]

        self._set_to_default_values()

    def _set_to_default_values(self):
        self._n_frames_passed = -1

    def reset(self):
        self._set_to_default_values()

    def run(self, marker_id_to_detections, current_frame_id):
        self._n_frames_passed += 1
        if self._n_frames_passed >= self._select_key_markers_interval:
            self._n_frames_passed = -1
            if self._check_if_key_marker(marker_id_to_detections):
                self.save_key_markers(marker_id_to_detections, current_frame_id)

    def _check_if_key_marker(self, marker_id_to_detections):
        if len(marker_id_to_detections) < self._min_n_markers_per_frame:
            return False

        if self._check_key_markers_bins_availability(marker_id_to_detections):
            return True
        else:
            return False

    def _check_key_markers_bins_availability(self, marker_id_to_detections):
        for marker_id, detection in marker_id_to_detections.items():
            n_same_markers_in_bin = len(
                [
                    marker
                    for marker in self._controller_storage.all_key_markers
                    if marker.marker_id == marker_id
                    and marker.bin == self._get_bin(detection)
                ]
            )
            if n_same_markers_in_bin < self._max_n_same_markers_per_bin:
                return True

        return False

    def save_key_markers(self, marker_id_to_detections, current_frame_id):
        key_markers = [
            KeyMarker(
                current_frame_id,
                marker_id,
                detection["verts"],
                self._get_bin(detection),
            )
            for marker_id, detection in marker_id_to_detections.items()
        ]
        self._controller_storage.all_key_markers += key_markers

        marker_ids = [marker.marker_id for marker in key_markers]
        key_edges = [
            (marker_id1, marker_id2, current_frame_id)
            for marker_id1, marker_id2 in list(it.combinations(marker_ids, 2))
        ]

        self._controller_storage.visibility_graph.add_edges_from(key_edges)

    def _get_bin(self, detection):
        centroid = detection["centroid"]
        bin_x = int(np.digitize(centroid[0], self._bins_x))
        bin_y = int(np.digitize(centroid[1], self._bins_y))
        return bin_x, bin_y
