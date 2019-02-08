import collections
import itertools as it
import logging

import numpy as np

logger = logging.getLogger(__name__)

KeyMarker = collections.namedtuple(
    "KeyMarker", ["frame_id", "marker_id", "verts", "bin"]
)


class VisibilityGraphs:
    def __init__(
        self,
        model_storage,
        select_key_markers_interval=5,
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

        self._n_bins_x = 8
        self._n_bins_y = 5
        self._bins_x = np.linspace(0, 1, self._n_bins_x + 1)[1:-1]
        self._bins_y = np.linspace(0, 1, self._n_bins_y + 1)[1:-1]

        self._set_to_default_values()

    def _set_to_default_values(self):
        self._n_frames_passed = 0

    def reset(self):
        self._set_to_default_values()

    def check_key_markers(self, marker_id_to_detections, current_frame_id):
        if self._n_frames_passed >= self._select_key_markers_interval:
            self._n_frames_passed = 0
            key_markers = self._pick_key_markers(
                marker_id_to_detections, current_frame_id
            )
            self._add_key_markers_to_model_storage(key_markers, current_frame_id)

        self._n_frames_passed += 1

    def _pick_key_markers(self, marker_id_to_detections, current_frame_id):
        if len(marker_id_to_detections) < self._min_n_markers_per_frame:
            return []

        key_marker_candidates = self._get_key_marker_candidates(
            marker_id_to_detections, current_frame_id
        )

        # if there are markers which have not yet been optimized,
        # add all markers in this frame and
        # do not need to check if the corresponding bins are available
        if not bool(
            marker_id_to_detections.keys()
            - self._model_storage.marker_id_to_extrinsics_opt.keys()
        ):
            key_marker_candidates = self._filter_key_markers_by_bins_availability(
                key_marker_candidates
            )

        if len(key_marker_candidates) < self._min_n_markers_per_frame:
            return []

        return key_marker_candidates

    def _get_key_marker_candidates(self, marker_id_to_detections, current_frame_id):
        bins_x, bins_y = self._get_bins(marker_id_to_detections)
        key_marker_candidates = [
            KeyMarker(
                current_frame_id,
                marker_id,
                marker_id_to_detections[marker_id]["verts"],
                (x, y),
            )
            for marker_id, x, y in zip(marker_id_to_detections.keys(), bins_x, bins_y)
        ]
        return key_marker_candidates

    def _get_bins(self, marker_id_to_detections):
        centroids = np.array(
            [
                marker_id_to_detections[k]["centroid"]
                for k in marker_id_to_detections.keys()
            ]
        )
        bins_x = np.digitize(centroids[:, 0], self._bins_x)
        bins_y = np.digitize(centroids[:, 1], self._bins_y)
        return bins_x, bins_y

    def _filter_key_markers_by_bins_availability(self, key_marker_candidates):
        key_markers = []
        for candidate in key_marker_candidates:
            n_same_markers_in_bin = len(
                [
                    marker
                    for marker in self._model_storage.all_key_markers
                    if marker.marker_id == candidate.marker_id
                    and marker.bin == candidate.bin
                ]
            )
            if n_same_markers_in_bin < self._max_n_same_markers_per_bin:
                key_markers.append(candidate)

        return key_markers

    def _add_key_markers_to_model_storage(self, key_markers, current_frame_id):
        if not key_markers:
            return

        all_markers = [marker.marker_id for marker in key_markers]
        logger.debug(
            "frame {0} key_markers {1}".format(key_markers[0].frame_id, all_markers)
        )
        # the node of visibility_graph: marker_id;
        # the edge of visibility_graph: current_frame_id
        for u, v in list(it.combinations(all_markers, 2)):
            self._model_storage.visibility_graph.add_edge(u, v, key=current_frame_id)

        self._model_storage.all_key_markers += key_markers
        self._model_storage.n_new_key_markers_added += len(key_markers)
