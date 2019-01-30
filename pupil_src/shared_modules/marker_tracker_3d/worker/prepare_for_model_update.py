import collections
import logging

import networkx as nx

from observable import Observable

logger = logging.getLogger(__name__)


DataForModelInit = collections.namedtuple(
    "DataForModelInit",
    [
        "all_novel_markers",
        "frame_id_to_extrinsics_prv",
        "marker_id_to_extrinsics_prv",
        "frame_ids_to_be_optimized",
        "marker_ids_to_be_optimized",
    ],
)


class PrepareForModelUpdate(Observable):
    def __init__(
        self,
        model_storage,
        predetermined_origin_marker_id=None,
        min_n_novel_markers=5,
        min_n_frames_per_marker=2,
        min_n_markers_per_frame=2,
    ):
        assert min_n_markers_per_frame >= 2
        assert min_n_novel_markers >= 1
        assert min_n_frames_per_marker >= 2

        self._model_storage = model_storage
        self._predetermined_origin_marker_id = predetermined_origin_marker_id
        self._min_n_markers_per_frame = min_n_markers_per_frame
        self._min_n_novel_markers = min_n_novel_markers
        self._min_n_frames_per_marker = min_n_frames_per_marker

    def on_prepare_for_model_init_done(self, data_for_model_init):
        pass

    def run(self):
        if self._model_storage.model_being_updated:
            return

        # Do optimization only when there are enough new novel_markers selected
        if self._model_storage.n_new_novel_markers_added < self._min_n_novel_markers:
            return

        self._prepare_for_model_init()

    def _prepare_for_model_init(self):
        marker_ids_to_be_optimized = self._get_marker_ids_to_be_optimized()
        frame_ids_to_be_optimized = self._get_frame_ids_to_be_optimized(
            marker_ids_to_be_optimized
        )

        if frame_ids_to_be_optimized:
            self._model_storage.n_new_novel_markers_added = 0

            data_for_model_init = DataForModelInit(
                self._model_storage.all_novel_markers,
                self._model_storage.frame_id_to_extrinsics_opt,
                self._model_storage.marker_id_to_extrinsics_opt,
                frame_ids_to_be_optimized,
                marker_ids_to_be_optimized,
            )
            self.on_prepare_for_model_init_done(data_for_model_init)

    def _get_marker_ids_to_be_optimized(self):
        marker_id_candidates = self._filter_marker_ids_by_visibility_graph()
        marker_ids_to_be_optimized = [self._model_storage.origin_marker_id] + list(
            marker_id_candidates - {self._model_storage.origin_marker_id}
        )
        logger.debug(
            "marker_ids_to_be_optimized updated {}".format(marker_ids_to_be_optimized)
        )
        return marker_ids_to_be_optimized

    def _filter_marker_ids_by_visibility_graph(self):
        markers_enough_viewed = set(
            node
            for node in self._model_storage.visibility_graph.nodes
            if len(
                [
                    marker
                    for marker in self._model_storage.all_novel_markers
                    if marker.marker_id == node
                ]
            )
            >= self._min_n_frames_per_marker
        )
        try:
            markers_connected_to_first_marker = set(
                nx.node_connected_component(
                    self._model_storage.visibility_graph,
                    self._model_storage.origin_marker_id,
                )
            )
        except KeyError:
            # when origin_marker_id not in visibility_graph
            if self._model_storage.origin_marker_id is None:
                self._set_coordinate_system(markers_enough_viewed)
            return set()
        else:
            return markers_enough_viewed & markers_connected_to_first_marker

    def _set_coordinate_system(self, markers_enough_viewed):
        origin_marker_id = self._determine_origin_marker_id(markers_enough_viewed)
        self._model_storage.setup_origin_marker_id(origin_marker_id)

    def _determine_origin_marker_id(self, markers_enough_viewed):
        if self._predetermined_origin_marker_id is not None:
            origin_marker_id = self._predetermined_origin_marker_id
        else:
            try:
                origin_marker_id = list(markers_enough_viewed)[0]
            except IndexError:
                origin_marker_id = None

        return origin_marker_id

    def _get_frame_ids_to_be_optimized(self, marker_ids_to_be_optimized):
        frame_id_candidates = set(
            marker_candidate.frame_id
            for marker_candidate in self._model_storage.all_novel_markers
        )

        frame_ids_to_be_optimized = []
        for frame_id in frame_id_candidates:
            optimized_markers_in_frame = set(
                marker.marker_id
                for marker in self._model_storage.all_novel_markers
                if marker.frame_id == frame_id
                and marker.marker_id in marker_ids_to_be_optimized
            )
            if len(optimized_markers_in_frame) >= self._min_n_markers_per_frame:
                frame_ids_to_be_optimized.append(frame_id)

        logger.debug(
            "frame_ids_to_be_optimized updated {}".format(frame_ids_to_be_optimized)
        )
        return frame_ids_to_be_optimized
