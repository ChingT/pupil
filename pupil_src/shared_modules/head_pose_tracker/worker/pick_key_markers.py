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

        self._n_frames_passed = 0

    def run(self, markers_in_frame):
        if self._decide_key_markers(markers_in_frame):
            self._save_key_markers(markers_in_frame)

    def _decide_key_markers(self, markers_in_frame):
        self._n_frames_passed += 1
        if self._n_frames_passed >= self._select_key_markers_interval:
            self._n_frames_passed = 0

            if len(
                markers_in_frame
            ) >= self._min_n_markers_per_frame and self._check_bins_availability(
                markers_in_frame
            ):
                return True

        return False

    def _check_bins_availability(self, markers_in_frame):
        for marker in markers_in_frame:
            n_same_markers_in_bin = len(
                [
                    key_marker
                    for key_marker in self._optimization_storage.all_key_markers
                    if key_marker.marker_id == marker["id"]
                    and key_marker.bin == self._get_bin(marker)
                ]
            )
            if n_same_markers_in_bin < self._max_n_same_markers_per_bin:
                return True

        return False

    def _save_key_markers(self, markers_in_frame):
        key_markers = [
            KeyMarker(
                marker["timestamp"],
                marker["id"],
                marker["verts"],
                self._get_bin(marker),
            )
            for marker in markers_in_frame
        ]
        self._optimization_storage.all_key_markers += key_markers

    def _get_bin(self, detection):
        centroid = detection["centroid"]
        bin_x = int(np.digitize(centroid[0], self._bins_x))
        bin_y = int(np.digitize(centroid[1], self._bins_y))
        return bin_x, bin_y
