"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


class DecideKeyMarkers:
    def __init__(
        self,
        model_storage,
        select_key_markers_interval=1,
        min_n_markers_per_frame=2,
        max_n_same_markers_per_bin=1,
    ):
        assert select_key_markers_interval >= 1
        assert min_n_markers_per_frame >= 2
        assert max_n_same_markers_per_bin >= 1

        self._model_storage = model_storage
        self._select_key_markers_interval = select_key_markers_interval
        self._min_n_markers_per_frame = min_n_markers_per_frame
        self._max_n_same_markers_per_bin = max_n_same_markers_per_bin

        self._n_frames_passed = -1

    def run(self, marker_id_to_detections):
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
                    for marker in self._model_storage.all_key_markers
                    if marker.marker_id == marker_id
                    and marker.bin == self._model_storage.get_bin(detection)
                ]
            )
            if n_same_markers_in_bin < self._max_n_same_markers_per_bin:
                return True

        return False
