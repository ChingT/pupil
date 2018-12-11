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
        storage,
        camera_model,
        origin_marker_id=None,
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

        self.storage = storage
        self.camera_model = camera_model
        self._origin_marker_id = origin_marker_id
        self._select_novel_markers_interval = select_novel_markers_interval
        self._min_n_markers_per_frame = min_n_markers_per_frame
        self._max_n_same_markers_per_bin = max_n_same_markers_per_bin
        self._optimization_interval = optimization_interval
        self._min_n_frames_per_marker = min_n_frames_per_marker

        self._n_bins_x = 10
        self._n_bins_y = 6
        self._bins_x = np.linspace(0, 1, self._n_bins_x + 1)[1:-1]
        self._bins_y = np.linspace(0, 1, self._n_bins_y + 1)[1:-1]

        self._n_frames_passed = 0
        self._n_new_novel_markers_added = 0
        self.adding_marker_detections = True
        self.visibility_graph = nx.MultiGraph()

    def reset(self):
        self._n_frames_passed = 0
        self._n_new_novel_markers_added = 0
        self.adding_marker_detections = True
        self.visibility_graph = nx.MultiGraph()

        self.on_update_menu()

    def on_update_menu(self):
        pass

    def on_novel_markers_added(self):
        pass

    def on_ready_for_optimization(self):
        pass

    def add_observer_to_novel_markers_added(self):
        self.add_observer("on_novel_markers_added", self._prepare_for_optimization)

    def remove_observer_from_novel_markers_added(self):
        self.remove_observer("on_novel_markers_added", self._prepare_for_optimization)

    def add_observations(self, marker_detections, camera_extrinsics):
        self._save_current_camera_extrinsics(camera_extrinsics)

        if self.adding_marker_detections:
            if self._n_frames_passed >= self._select_novel_markers_interval:
                self._n_frames_passed = 0
                self._select_novel_markers(marker_detections)

        self.storage.current_frame_id += 1
        self._n_frames_passed += 1

    def _save_current_camera_extrinsics(self, camera_extrinsics):
        if camera_extrinsics is not None:
            self.storage.camera_extrinsics_opt[
                self.storage.current_frame_id
            ] = camera_extrinsics

    def _select_novel_markers(self, marker_detections):
        novel_markers = self._filter_novel_markers(marker_detections)

        if novel_markers:
            self._n_new_novel_markers_added += 1
            self._add_to_all_novel_markers(novel_markers)
            self.on_novel_markers_added()

    def _filter_novel_markers(self, marker_detections):
        if len(marker_detections) < self._min_n_markers_per_frame:
            return []

        novel_marker_candidates = self._get_novel_marker_candidates(marker_detections)

        # if there are markers which have not yet been optimized,
        # add all markers in this frame and
        # do not need to check if the corresponding bins are available
        if not bool(
            marker_detections.keys() - self.storage.marker_extrinsics_opt.keys()
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
                frame_id=self.storage.current_frame_id,
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
                    for marker in self.storage.all_novel_markers
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
            self.visibility_graph.add_edge(u, v, key=self.storage.current_frame_id)

        self.storage.all_novel_markers += novel_markers

    def _prepare_for_optimization(self):
        # Do optimization when there are some new novel_markers selected
        if self._n_new_novel_markers_added >= self._optimization_interval:
            self._n_new_novel_markers_added = 0

            self._update_markers_id()
            self._update_frames_id()
            self.on_ready_for_optimization()

    def _update_markers_id(self):
        marker_id_candidates = self._filter_markers_id_by_visibility_graph()

        try:
            markers_id = [self.storage.markers_id[0]] + [
                marker_id
                for marker_id in marker_id_candidates
                if marker_id != self.storage.markers_id[0]
            ]
        except IndexError:
            return

        self.storage.markers_id = markers_id
        logger.debug("markers_id updated {}".format(self.storage.markers_id))

    def _filter_markers_id_by_visibility_graph(self):
        markers_enough_viewed = set(
            node
            for node in self.visibility_graph.nodes
            if len(
                [
                    marker
                    for marker in self.storage.all_novel_markers
                    if marker.marker_id == node
                ]
            )
            >= self._min_n_frames_per_marker
        )
        try:
            markers_connected_to_first_marker = set(
                nx.node_connected_component(
                    self.visibility_graph, self.storage.markers_id[0]
                )
            )
        except IndexError:
            # when self.storage.markers_id == []
            self._set_coordinate_system(markers_enough_viewed)
            return set()
        except KeyError:
            # self.storage.markers_id[0] not in visibility_graph
            return set()
        else:
            return markers_enough_viewed & markers_connected_to_first_marker

    def _set_coordinate_system(self, markers_enough_viewed):
        if self._origin_marker_id:
            origin_marker_id = self._origin_marker_id
        else:
            try:
                origin_marker_id = list(markers_enough_viewed)[0]
            except IndexError:
                return

        self.storage.markers_id = [origin_marker_id]
        self.storage.marker_extrinsics_opt = {
            origin_marker_id: utils.marker_extrinsics_origin
        }
        self.storage.marker_points_3d_opt = {origin_marker_id: utils.marker_df}

        self.on_update_menu()

    def _update_frames_id(self):
        frames_id = []
        frame_id_candidates = set(
            marker_candidate.frame_id
            for marker_candidate in self.storage.all_novel_markers
        )

        for frame_id in frame_id_candidates:
            optimized_markers_in_frame = set(
                marker.marker_id
                for marker in self.storage.all_novel_markers
                if marker.frame_id == frame_id
                and marker.marker_id in self.storage.markers_id
            )
            if len(optimized_markers_in_frame) >= self._min_n_markers_per_frame:
                frames_id.append(frame_id)

        self.storage.frames_id = sorted(frames_id)
        logger.debug("frames_id updated {}".format(self.storage.frames_id))

    def process_optimization_results(self, optimization_result):
        """ process the results of optimization; update camera_extrinsics_opt,
        marker_extrinsics_opt and marker_points_3d_opt """

        if optimization_result:
            self._update_extrinsics_opt(optimization_result)
            self.discard_failed_frames(optimization_result)
        self.add_observer_to_novel_markers_added()

    def _update_extrinsics_opt(self, optimization_result):
        for i, p in enumerate(optimization_result.camera_extrinsics_opt):
            self.storage.camera_extrinsics_opt[self.storage.frames_id[i]] = p

        for i, p in enumerate(optimization_result.marker_extrinsics_opt):
            if i not in optimization_result.marker_indices_failed:
                self.storage.marker_extrinsics_opt[self.storage.markers_id[i]] = p
                self.storage.marker_points_3d_opt[
                    self.storage.markers_id[i]
                ] = utils.params_to_points_3d(p)[0]

        logger.info(
            "{0} markers have been registered and updated {1} {2}".format(
                len(self.storage.marker_extrinsics_opt),
                len(optimization_result.marker_indices_failed),
                len(optimization_result.marker_extrinsics_opt),
            )
        )

    def discard_failed_frames(self, optimization_result):
        frames_id_failed = list(
            self.storage.frames_id[i] for i in optimization_result.frame_indices_failed
        )
        logger.debug("discard_failed_frames {0}".format(frames_id_failed))

        if frames_id_failed:
            redundant_edges = [
                (node, neighbor, frame_id)
                for node, neighbor, frame_id in self.visibility_graph.edges(keys=True)
                if frame_id in frames_id_failed
            ]
            self.visibility_graph.remove_edges_from(redundant_edges)

            self.storage.all_novel_markers = [
                marker
                for marker in self.storage.all_novel_markers
                if marker.frame_id not in frames_id_failed
            ]

    # For debug TODO: remove save_graph()
    def save_graph(self, save_path):
        import matplotlib.pyplot as plt
        import os

        if self.visibility_graph and self.storage.markers_id:
            graph_vis = self.visibility_graph.copy()
            all_nodes = list(graph_vis.nodes)

            pos = nx.spring_layout(graph_vis, seed=0)  # positions for all nodes
            pos_label = dict((n, pos[n] + 0.05) for n in pos)

            nx.draw_networkx_nodes(
                graph_vis, pos, nodelist=all_nodes, node_color="g", node_size=100
            )
            nx.draw_networkx_nodes(
                graph_vis,
                pos,
                nodelist=self.storage.markers_id,
                node_color="r",
                node_size=100,
            )
            nx.draw_networkx_edges(graph_vis, pos, width=1, alpha=0.1)
            nx.draw_networkx_labels(graph_vis, pos, font_size=7)

            labels = dict(
                (
                    n,
                    self.storage.markers_id.index(n)
                    if n in self.storage.markers_id
                    else None,
                )
                for n in graph_vis.nodes()
            )
            nx.draw_networkx_labels(
                graph_vis, pos=pos_label, labels=labels, font_size=6, font_color="b"
            )

            plt.axis("off")
            save_name = os.path.join(
                save_path,
                "visibility_graph-{0}-{1}.png".format(
                    len(self.visibility_graph), len(self.storage.markers_id)
                ),
            )
            plt.savefig(save_name)
            plt.clf()
