import collections
import itertools as it
import logging

import numpy as np

from observable import Observable

logger = logging.getLogger(__name__)

NovelMarker = collections.namedtuple(
    "NovelMarker", ["frame_id", "marker_id", "verts", "bin"]
)


class VisibilityGraphs(Observable):
    def __init__(
        self,
        model_storage,
        select_novel_markers_interval=5,
        min_n_markers_per_frame=2,
        max_n_same_markers_per_bin=1,
    ):
        assert select_novel_markers_interval >= 1
        assert min_n_markers_per_frame >= 2
        assert max_n_same_markers_per_bin >= 1

        self._model_storage = model_storage
        self._select_novel_markers_interval = select_novel_markers_interval
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

    def on_novel_markers_added(self):
        pass

    def check_novel_markers(self, marker_id_to_detections, current_frame_id):
        if not self._model_storage.adding_observations:
            return False

        if self._n_frames_passed >= self._select_novel_markers_interval:
            self._n_frames_passed = 0
            novel_markers = self._pick_novel_markers(
                marker_id_to_detections, current_frame_id
            )
            self._add_novel_markers_to_model_storage(novel_markers, current_frame_id)
        else:
            novel_markers = []

        self._n_frames_passed += 1

        if novel_markers:
            return True
        else:
            return False

    def _pick_novel_markers(self, marker_id_to_detections, current_frame_id):
        if len(marker_id_to_detections) < self._min_n_markers_per_frame:
            return []

        novel_marker_candidates = self._get_novel_marker_candidates(
            marker_id_to_detections, current_frame_id
        )

        # if there are markers which have not yet been optimized,
        # add all markers in this frame and
        # do not need to check if the corresponding bins are available
        if not bool(
            marker_id_to_detections.keys()
            - self._model_storage.marker_id_to_extrinsics_opt.keys()
        ):
            novel_marker_candidates = self._filter_novel_markers_by_bins_availability(
                novel_marker_candidates
            )

        if len(novel_marker_candidates) < self._min_n_markers_per_frame:
            return []

        return novel_marker_candidates

    def _get_novel_marker_candidates(self, marker_id_to_detections, current_frame_id):
        bins_x, bins_y = self._get_bins(marker_id_to_detections)
        novel_marker_candidates = [
            NovelMarker(
                current_frame_id,
                marker_id,
                marker_id_to_detections[marker_id]["verts"],
                (x, y),
            )
            for marker_id, x, y in zip(marker_id_to_detections.keys(), bins_x, bins_y)
        ]
        return novel_marker_candidates

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

    def _filter_novel_markers_by_bins_availability(self, novel_marker_candidates):
        novel_markers = []
        for candidate in novel_marker_candidates:
            n_same_markers_in_bin = len(
                [
                    marker
                    for marker in self._model_storage.all_novel_markers
                    if marker.marker_id == candidate.marker_id
                    and marker.bin == candidate.bin
                ]
            )
            if n_same_markers_in_bin < self._max_n_same_markers_per_bin:
                novel_markers.append(candidate)

        return novel_markers

    def _add_novel_markers_to_model_storage(self, novel_markers, current_frame_id):
        if not novel_markers:
            return

        all_markers = [marker.marker_id for marker in novel_markers]
        logger.debug(
            "frame {0} novel_markers {1}".format(novel_markers[0].frame_id, all_markers)
        )
        # the node of visibility_graph: marker_id;
        # the edge of visibility_graph: current_frame_id
        for u, v in list(it.combinations(all_markers, 2)):
            self._model_storage.visibility_graph.add_edge(u, v, key=current_frame_id)

        self._model_storage.all_novel_markers += novel_markers
        self._model_storage.n_new_novel_markers_added += len(novel_markers)
        self.on_novel_markers_added()
