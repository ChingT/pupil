import collections
import itertools as it
import logging

import networkx as nx
import numpy as np

from marker_tracker_3d import utils
from observable import Observable

logger = logging.getLogger(__name__)

MarkerCandidate = collections.namedtuple("MarkerCandidate", ["id", "bin"])


class VisibilityGraphs(Observable):
    def __init__(
        self,
        storage,
        camera_model,
        origin_marker_id=None,
        select_keyframe_interval=6,
        min_n_markers_per_frame=2,
        max_n_same_markers_per_bin=1,
        optimization_interval=1,
        min_n_frames_per_marker=2,
    ):
        assert select_keyframe_interval >= 1
        assert min_n_markers_per_frame >= 2
        assert max_n_same_markers_per_bin >= 1
        assert optimization_interval >= 1
        assert min_n_frames_per_marker >= 2

        self.storage = storage
        self.camera_model = camera_model
        self._origin_marker_id = origin_marker_id
        self._select_keyframe_interval = select_keyframe_interval
        self._min_n_markers_per_frame = min_n_markers_per_frame
        self._max_n_same_markers_per_bin = max_n_same_markers_per_bin
        self._optimization_interval = optimization_interval
        self._min_n_frames_per_marker = min_n_frames_per_marker

        self._n_bins_x = 10
        self._n_bins_y = 6
        self._bins_x = np.linspace(0, 1, self._n_bins_x + 1)[1:-1]
        self._bins_y = np.linspace(0, 1, self._n_bins_y + 1)[1:-1]

        self._all_marker_location = {
            (x, y): {} for x in range(self._n_bins_x) for y in range(self._n_bins_y)
        }
        self._n_frames_passed = 0
        self._n_new_keyframe_added = 0
        self.adding_marker_detections = True
        self.visibility_graph = nx.MultiGraph()

    def reset(self):
        self._all_marker_location = {
            (x, y): {} for x in range(self._n_bins_x) for y in range(self._n_bins_y)
        }
        self._n_frames_passed = 0
        self._n_new_keyframe_added = 0
        self.adding_marker_detections = True
        self.visibility_graph = nx.MultiGraph()

        self.on_update_menu()

    def on_update_menu(self):
        pass

    def on_keyframe_added(self):
        pass

    def on_ready_for_optimization(self):
        pass

    def add_observer_to_keyframe_added(self):
        self.add_observer("on_keyframe_added", self._prepare_for_optimization)

    def remove_observer_from_keyframe_added(self):
        self.remove_observer("on_keyframe_added", self._prepare_for_optimization)

    def add_marker_detections(self, marker_detections):
        if self._n_frames_passed >= self._select_keyframe_interval:
            self._n_frames_passed = 0
            if self.adding_marker_detections:
                self._select_keyframe(marker_detections)

        self._n_frames_passed += 1

    def _select_keyframe(self, marker_detections):
        novel_marker_detections = self._filter_novel_marker_detections(
            marker_detections
        )

        if novel_marker_detections:
            self._n_new_keyframe_added += 1
            self._add_to_keyframes(novel_marker_detections)
            self._add_to_visibility_graph(novel_marker_detections)
            self.on_keyframe_added()

    def _filter_novel_marker_detections(self, marker_detections):
        marker_candidates = self._get_marker_candidates(marker_detections)
        self._add_to_all_marker_location(marker_candidates)

        novel_marker_detections = {
            marker.id: marker_detections[marker.id] for marker in marker_candidates
        }
        return novel_marker_detections

    def _get_marker_candidates(self, marker_detections):
        if len(marker_detections) < self._min_n_markers_per_frame:
            return []

        bins_x, bins_y = self._get_bins(marker_detections)
        no_need_to_check_bin = bool(marker_detections.keys() - self.storage.marker_keys)
        # if there are markers which have not yet been optimized,
        # add all markers in this frame and
        # do not need to check if the corresponding bins are available
        marker_candidates = [
            MarkerCandidate(id=marker_id, bin=(x, y))
            for marker_id, x, y in zip(marker_detections.keys(), bins_x, bins_y)
            if no_need_to_check_bin or self._check_bin_available(marker_id, (x, y))
        ]

        if len(marker_candidates) < self._min_n_markers_per_frame:
            return []

        return marker_candidates

    def _get_bins(self, marker_detections):
        all_centroids = np.array(
            [marker_detections[k]["centroid"] for k in marker_detections.keys()]
        )
        bins_x = np.digitize(all_centroids[:, 0], self._bins_x)
        bins_y = np.digitize(all_centroids[:, 1], self._bins_y)
        return bins_x, bins_y

    def _check_bin_available(self, marker_id, marker_bin):
        try:
            count = len(self._all_marker_location[marker_bin][marker_id])
        except KeyError:
            return True

        return count < self._max_n_same_markers_per_bin

    def _add_to_all_marker_location(self, marker_candidates):
        for marker in marker_candidates:
            try:
                self._all_marker_location[marker.bin][marker.id].append(
                    self.storage.frame_id
                )
            except KeyError:
                self._all_marker_location[marker.bin][marker.id] = [
                    self.storage.frame_id
                ]

    def _add_to_keyframes(self, novel_marker_detections):
        self.storage.keyframes[self.storage.frame_id] = novel_marker_detections
        logger.debug(
            "--> keyframe {0}; markers {1}".format(
                self.storage.frame_id, list(novel_marker_detections.keys())
            )
        )

    def _add_to_visibility_graph(self, novel_marker_detections):
        """
        the node of visibility_graph: marker id; attributes: the keyframe id
        the edge of visibility_graph: keyframe id, where two markers shown in the same frame
        """

        # add frame_id as edges in the graph
        for u, v in list(it.combinations(novel_marker_detections.keys(), 2)):
            self.visibility_graph.add_edge(u, v, key=self.storage.frame_id)

        # add frame_id as an attribute of the node
        for marker_id in novel_marker_detections.keys():
            self.visibility_graph.nodes[marker_id][self.storage.frame_id] = []

    def _prepare_for_optimization(self):
        # Do optimization when there are some new keyframes selected
        if self._n_new_keyframe_added >= self._optimization_interval:
            self._n_new_keyframe_added = 0

            self._update_marker_keys()
            self._update_camera_keys()
            self.on_ready_for_optimization()

    def _update_marker_keys(self):
        candidate_nodes = self._filter_candidate_nodes()
        for node in candidate_nodes:
            if node not in self.storage.marker_keys:
                self.storage.marker_keys.append(node)

        logger.debug("marker_keys updated {}".format(self.storage.marker_keys))

    def _update_camera_keys(self):
        for f_id, frame in self.storage.keyframes.items():
            if (
                f_id not in self.storage.camera_keys
                and len(frame.keys() & self.storage.marker_keys)
                >= self._min_n_markers_per_frame
            ):
                self.storage.camera_keys.append(f_id)

        logger.debug("camera_keys updated {}".format(sorted(self.storage.camera_keys)))

    def _filter_candidate_nodes(self):
        nodes_enough_viewed = set(
            node
            for node in self.visibility_graph.nodes
            if len(self.visibility_graph.nodes[node]) >= self._min_n_frames_per_marker
        )
        try:
            nodes_connected_to_first_node = set(
                nx.node_connected_component(
                    self.visibility_graph, self.storage.marker_keys[0]
                )
            )
        except IndexError:
            # when self.storage.marker_keys == []
            self._set_coordinate_system(nodes_enough_viewed)
            return set()
        except KeyError:
            # self.storage.marker_keys[0] not in visibility_graph
            return set()
        else:
            return nodes_enough_viewed & nodes_connected_to_first_node

    def _set_coordinate_system(self, nodes_enough_viewed):
        if self._origin_marker_id:
            origin_marker_id = self._origin_marker_id
        else:
            try:
                origin_marker_id = list(nodes_enough_viewed)[0]
            except IndexError:
                return

        self.storage.marker_keys = [origin_marker_id]
        self.storage.marker_extrinsics_opt = {
            origin_marker_id: utils.marker_extrinsics_origin
        }
        self.storage.marker_points_3d_opt = {origin_marker_id: utils.marker_df}

        self.on_update_menu()

    # For debug TODO: remove save_graph()
    def save_graph(self, save_path):
        import matplotlib.pyplot as plt
        import os

        if self.visibility_graph and self.storage.marker_keys:
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
                nodelist=self.storage.marker_keys,
                node_color="r",
                node_size=100,
            )
            nx.draw_networkx_edges(graph_vis, pos, width=1, alpha=0.1)
            nx.draw_networkx_labels(graph_vis, pos, font_size=7)

            labels = dict(
                (
                    n,
                    self.storage.marker_keys.index(n)
                    if n in self.storage.marker_keys
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
                    len(self.visibility_graph), len(self.storage.marker_keys)
                ),
            )
            plt.savefig(save_name)
            plt.clf()
