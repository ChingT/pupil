import logging

logger = logging.getLogger(__name__)


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

        self._set_to_default_values()

    def _set_to_default_values(self):
        self._n_frames_passed = -1

    def reset(self):
        self._set_to_default_values()

    def run(self, marker_id_to_detections):
        self._n_frames_passed += 1
        if self._n_frames_passed >= self._select_key_markers_interval:
            self._n_frames_passed = -1
            return self._check_if_key_marker(marker_id_to_detections)
        else:
            return False

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
                    marker_id
                    for _, key_marker_id in self._controller_storage.key_markers_bins[
                        detection["bin"]
                    ]
                    if marker_id == key_marker_id
                ]
            )
            if n_same_markers_in_bin < self._max_n_same_markers_per_bin:
                return True

        return False
