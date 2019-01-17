import collections
import itertools as it
import logging

import networkx as nx
import numpy as np

from marker_tracker_3d import utils
from observable import Observable

logger = logging.getLogger(__name__)

NovelMarker = collections.namedtuple(
    "NovelMarker", ["frame_id", "marker_id", "verts", "bin"]
)


class VisibilityGraphs(Observable):
    def __init__(
        self,
        model_optimization_storage,
        camera_model,
        predetermined_origin_marker_id=None,
        select_novel_markers_interval=6,
        min_n_markers_per_frame=2,
        max_n_same_markers_per_bin=1,
        optimization_interval=1,
        min_n_frames_per_marker=2,
    ):
        assert select_novel_markers_interval >= 1
        assert min_n_markers_per_frame >= 2
        assert max_n_same_markers_per_bin >= 1
        assert optimization_interval >= 1
        assert min_n_frames_per_marker >= 2

        self._model_optimization_storage = model_optimization_storage
        self._camera_model = camera_model
        self._predetermined_origin_marker_id = predetermined_origin_marker_id
        self._select_novel_markers_interval = select_novel_markers_interval
        self._min_n_markers_per_frame = min_n_markers_per_frame
        self._max_n_same_markers_per_bin = max_n_same_markers_per_bin
        self._optimization_interval = optimization_interval
        self._min_n_frames_per_marker = min_n_frames_per_marker

        self._n_bins_x = 10
        self._n_bins_y = 6
        self._bins_x = np.linspace(0, 1, self._n_bins_x + 1)[1:-1]
        self._bins_y = np.linspace(0, 1, self._n_bins_y + 1)[1:-1]

        self._set_to_default_values()

    def _set_to_default_values(self):
        self._origin_marker_id = None
        self._n_frames_passed = 0
        self._n_new_novel_markers_added = 0
        self._visibility_graph = nx.MultiGraph()

        self._optimization_requested = True

    def reset(self):
        self._set_to_default_values()

    def on_ready_for_optimization(self):
        pass

    def add_observations(self, marker_detections, camera_extrinsics):
        self._save_current_camera_extrinsics(camera_extrinsics)

        if self._model_optimization_storage.adding_marker_detections:
            if self._n_frames_passed >= self._select_novel_markers_interval:
                self._n_frames_passed = 0
                self._select_novel_markers(marker_detections)

        self._model_optimization_storage.current_frame_id += 1
        self._n_frames_passed += 1

    def _save_current_camera_extrinsics(self, camera_extrinsics):
        if camera_extrinsics is not None:
            self._model_optimization_storage.camera_extrinsics_opt_dict[
                self._model_optimization_storage.current_frame_id
            ] = camera_extrinsics

    def _select_novel_markers(self, marker_detections):
        novel_markers = self._filter_novel_markers(marker_detections)

        if novel_markers:
            self._add_to_all_novel_markers(novel_markers)
            self._n_new_novel_markers_added += 1

            self._prepare_for_optimization()

    def _filter_novel_markers(self, marker_detections):
        if len(marker_detections) < self._min_n_markers_per_frame:
            return []

        novel_marker_candidates = self._get_novel_marker_candidates(marker_detections)

        # if there are markers which have not yet been optimized,
        # add all markers in this frame and
        # do not need to check if the corresponding bins are available
        if not bool(
            marker_detections.keys()
            - self._model_optimization_storage.marker_extrinsics_opt_dict.keys()
        ):
            novel_marker_candidates = self._filter_novel_markers_by_bins_availability(
                novel_marker_candidates
            )

        if len(novel_marker_candidates) < self._min_n_markers_per_frame:
            return []

        return novel_marker_candidates

    def _get_novel_marker_candidates(self, marker_detections):
        bins_x, bins_y = self._get_bins(marker_detections)
        novel_marker_candidates = [
            NovelMarker(
                frame_id=self._model_optimization_storage.current_frame_id,
                marker_id=marker_id,
                verts=marker_detections[marker_id]["verts"],
                bin=(x, y),
            )
            for marker_id, x, y in zip(marker_detections.keys(), bins_x, bins_y)
        ]
        return novel_marker_candidates

    def _get_bins(self, marker_detections):
        centroids = np.array(
            [marker_detections[k]["centroid"] for k in marker_detections.keys()]
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
                    for marker in self._model_optimization_storage.all_novel_markers
                    if marker.marker_id == candidate.marker_id
                    and marker.bin == candidate.bin
                ]
            )
            if n_same_markers_in_bin < self._max_n_same_markers_per_bin:
                novel_markers.append(candidate)

        return novel_markers

    def _add_to_all_novel_markers(self, novel_markers):
        all_markers = [marker.marker_id for marker in novel_markers]
        logger.debug(
            "frame {0} novel_markers {1}".format(novel_markers[0].frame_id, all_markers)
        )
        # the node of visibility_graph: marker_id;
        # the edge of visibility_graph: current_frame_id
        for u, v in list(it.combinations(all_markers, 2)):
            self._visibility_graph.add_edge(
                u, v, key=self._model_optimization_storage.current_frame_id
            )

        self._model_optimization_storage.all_novel_markers += novel_markers

    def _prepare_for_optimization(self):
        # Do optimization when there are some new novel_markers selected
        if (
            self._optimization_requested
            and self._n_new_novel_markers_added >= self._optimization_interval
        ):
            self._optimization_requested = False
            self._n_new_novel_markers_added = 0

            self._update_marker_ids()
            self._update_frame_ids()

            self.on_ready_for_optimization()

    def _update_marker_ids(self):
        marker_id_candidates = self._filter_marker_ids_by_visibility_graph()

        try:
            marker_ids = [self._model_optimization_storage.marker_ids[0]] + [
                marker_id
                for marker_id in marker_id_candidates
                if marker_id != self._model_optimization_storage.marker_ids[0]
            ]
        except IndexError:
            pass
        else:
            self._model_optimization_storage.marker_ids = marker_ids
            logger.debug(
                "marker_ids updated {}".format(
                    self._model_optimization_storage.marker_ids
                )
            )

    def _filter_marker_ids_by_visibility_graph(self):
        markers_enough_viewed = set(
            node
            for node in self._visibility_graph.nodes
            if len(
                [
                    marker
                    for marker in self._model_optimization_storage.all_novel_markers
                    if marker.marker_id == node
                ]
            )
            >= self._min_n_frames_per_marker
        )
        try:
            markers_connected_to_first_marker = set(
                nx.node_connected_component(
                    self._visibility_graph,
                    self._model_optimization_storage.marker_ids[0],
                )
            )
        except IndexError:
            # when self.storage.marker_ids == []
            self._set_coordinate_system(markers_enough_viewed)
            return set()
        except KeyError:
            # self.storage.marker_ids[0] not in visibility_graph
            return set()
        else:
            return markers_enough_viewed & markers_connected_to_first_marker

    def _set_coordinate_system(self, markers_enough_viewed):
        self._origin_marker_id = self._find_origin_marker_id(markers_enough_viewed)
        if self._origin_marker_id is not None:
            self.set_up_origin_marker()

    def _find_origin_marker_id(self, markers_enough_viewed):
        if self._predetermined_origin_marker_id:
            origin_marker_id = self._predetermined_origin_marker_id
        else:
            try:
                origin_marker_id = list(markers_enough_viewed)[0]
            except IndexError:
                origin_marker_id = None

        return origin_marker_id

    def set_up_origin_marker(self):
        self._model_optimization_storage.marker_ids = [self._origin_marker_id]
        self._model_optimization_storage.marker_extrinsics_opt_dict = {
            self._origin_marker_id: utils.get_marker_extrinsics_origin()
        }
        self._model_optimization_storage.marker_points_3d_opt = {
            self._origin_marker_id: utils.get_marker_points_3d_origin()
        }

    def _update_frame_ids(self):
        frame_ids = []
        frame_id_candidates = set(
            marker_candidate.frame_id
            for marker_candidate in self._model_optimization_storage.all_novel_markers
        )

        for frame_id in frame_id_candidates:
            optimized_markers_in_frame = set(
                marker.marker_id
                for marker in self._model_optimization_storage.all_novel_markers
                if marker.frame_id == frame_id
                and marker.marker_id in self._model_optimization_storage.marker_ids
            )
            if len(optimized_markers_in_frame) >= self._min_n_markers_per_frame:
                frame_ids.append(frame_id)

        self._model_optimization_storage.frame_ids = sorted(frame_ids)
        logger.debug(
            "frame_ids updated {}".format(self._model_optimization_storage.frame_ids)
        )

    def process_optimization_results(self, optimization_result):
        """ process the results of optimization; update camera_extrinsics_opt_array,
        marker_extrinsics_opt_array and marker_points_3d_opt """

        if optimization_result:
            self._update_extrinsics_opt_array(optimization_result)
            self.discard_failed_frames(optimization_result)
        self._optimization_requested = True

    def _update_extrinsics_opt_array(self, optimization_result):
        for i, p in enumerate(optimization_result.camera_extrinsics_opt_array):
            self._model_optimization_storage.camera_extrinsics_opt_dict[
                self._model_optimization_storage.frame_ids[i]
            ] = p

        for i, p in enumerate(optimization_result.marker_extrinsics_opt_array):
            if i not in optimization_result.marker_indices_failed:
                self._model_optimization_storage.marker_extrinsics_opt_dict[
                    self._model_optimization_storage.marker_ids[i]
                ] = p
                self._model_optimization_storage.marker_points_3d_opt[
                    self._model_optimization_storage.marker_ids[i]
                ] = utils.extrinsics_to_marker_points_3d(p)[0]

        logger.debug(
            "{} markers have been registered and updated".format(
                len(self._model_optimization_storage.marker_extrinsics_opt_dict)
            )
        )

    def discard_failed_frames(self, optimization_result):
        frame_ids_failed = list(
            self._model_optimization_storage.frame_ids[i]
            for i in optimization_result.frame_indices_failed
        )
        logger.debug("discard_failed_frames {0}".format(frame_ids_failed))

        if frame_ids_failed:
            redundant_edges = [
                (node, neighbor, frame_id)
                for node, neighbor, frame_id in self._visibility_graph.edges(keys=True)
                if frame_id in frame_ids_failed
            ]
            self._visibility_graph.remove_edges_from(redundant_edges)

            self._model_optimization_storage.all_novel_markers = [
                marker
                for marker in self._model_optimization_storage.all_novel_markers
                if marker.frame_id not in frame_ids_failed
            ]
