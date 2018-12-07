import collections
import itertools as it
import logging

import networkx as nx
import numpy as np

from marker_tracker_3d import math
from marker_tracker_3d import utils
from observable import Observable

logger = logging.getLogger(__name__)


DataForOptimization = collections.namedtuple(
    "DataForOptimization",
    [
        "camera_indices",
        "marker_indices",
        "markers_points_2d_detected",
        "camera_extrinsics_prv",
        "marker_extrinsics_prv",
    ],
)
MarkerCandidate = collections.namedtuple("MarkerCandidate", ["id", "bin", "normal"])


class VisibilityGraphs(Observable):
    def __init__(
        self,
        storage,
        camera_model,
        origin_marker_id=None,
        select_keyframe_interval=10,
        min_n_markers_per_frame=2,
        max_n_same_markers_per_bin=5,
        optimization_interval=2,
        min_angle_diff=0.1,
        min_n_frames_per_marker=2,
    ):
        assert select_keyframe_interval >= 1
        assert min_n_markers_per_frame >= 2
        assert max_n_same_markers_per_bin >= 1
        assert optimization_interval >= 1
        assert min_angle_diff >= 0
        assert min_n_frames_per_marker >= 2

        self.storage = storage
        self.camera_model = camera_model
        self._origin_marker_id = origin_marker_id
        self._select_keyframe_interval = select_keyframe_interval
        self._min_n_markers_per_frame = min_n_markers_per_frame
        self._max_n_same_markers_per_bin = max_n_same_markers_per_bin
        self._optimization_interval = optimization_interval
        self._min_angle_diff = min_angle_diff
        self._min_n_frames_per_marker = min_n_frames_per_marker

        self._bins = np.linspace(0, 1, 10)

        self._all_marker_location = {(x, y): {} for x in range(10) for y in range(10)}
        self._n_frames_passed = 0
        self._n_new_keyframe_added = 0
        self.adding_marker_detections = True

    def reset(self):
        self._all_marker_location = {(x, y): {} for x in range(10) for y in range(10)}
        self._n_frames_passed = 0
        self._n_new_keyframe_added = 0
        self.adding_marker_detections = True

        self.on_update_menu()

    def on_update_menu(self):
        pass

    def on_keyframe_added(self):
        pass

    def on_data_for_optimization_prepared(self, data_for_optimization):
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
        """ select keyframe and update visibility_graph """
        if len(marker_detections) < self._min_n_markers_per_frame:
            return

        novel_marker_detections = self._filter_novel_marker_detections(
            marker_detections
        )

        if novel_marker_detections:
            self._n_new_keyframe_added += 1
            self._add_to_keyframes(novel_marker_detections)
            self._add_to_visibility_graph(novel_marker_detections)
            self.on_keyframe_added()

    def _filter_novel_marker_detections(self, marker_detections):
        marker_candidates = []
        for marker_id in marker_detections.keys():
            marker_bin = self._get_bin(marker_detections[marker_id])
            marker_normal = self._find_diverse_marker_normal(
                marker_id, marker_bin, marker_detections[marker_id]["verts"]
            )
            if marker_normal is None:
                continue
            marker_candidates.append(
                MarkerCandidate(marker_id, marker_bin, marker_normal)
            )

        if len(marker_candidates) < self._min_n_markers_per_frame:
            return

        self._add_to_all_marker_location(marker_candidates)
        novel_marker_detections = {
            marker.id: marker_detections[marker.id] for marker in marker_candidates
        }
        return novel_marker_detections

    def _get_bin(self, marker_detection):
        return tuple(np.digitize(marker_detection["centroid"], self._bins, right=True))

    def _find_diverse_marker_normal(self, marker_id, marker_bin, verts):
        """ if the bin is not full and the marker normal is diverse enough,
        return marker_normal
        """

        try:
            count = len(self._all_marker_location[marker_bin][marker_id])
        except KeyError:
            return self._compute_marker_normal(verts)

        if count < self._max_n_same_markers_per_bin:
            marker_normal = self._compute_marker_normal(verts)
            angle_diff = math.closest_angle_diff(
                marker_normal, self._all_marker_location[marker_bin][marker_id]
            )
            if angle_diff >= self._min_angle_diff:
                return marker_normal
        else:
            return

    def _compute_marker_normal(self, verts):
        retval, rvec, _ = self.camera_model.solvePnP(utils.marker_df, verts)
        marker_normal = rvec.ravel()
        return marker_normal

    def _add_to_all_marker_location(self, marker_candidates):
        for marker in marker_candidates:
            try:
                self._all_marker_location[marker.bin][marker.id].append(marker.normal)
            except KeyError:
                self._all_marker_location[marker.bin][marker.id] = [marker.normal]

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
            self.storage.visibility_graph.add_edge(u, v)

        # add frame_id as an attribute of the node
        for marker_id in novel_marker_detections.keys():
            self.storage.visibility_graph.nodes[marker_id][self.storage.frame_id] = []

    def _prepare_for_optimization(self):
        # Do optimization when there are some new keyframes selected
        if self._n_new_keyframe_added >= self._optimization_interval:
            self._n_new_keyframe_added = 0

            self._update_camera_and_marker_keys()
            self._collect_data_for_optimization()

    def _update_camera_and_marker_keys(self):
        candidate_nodes = self._filter_candidate_nodes()

        for node in candidate_nodes:
            if node not in self.storage.marker_keys:
                self.storage.marker_keys.append(node)

        for f_id, frame in self.storage.keyframes.items():
            if (
                f_id not in self.storage.camera_keys
                and len(frame.keys() & self.storage.marker_keys)
                >= self._min_n_markers_per_frame
            ):
                self.storage.camera_keys.append(f_id)

        logger.debug("marker_keys updated {}".format(self.storage.marker_keys))
        logger.debug("camera_keys updated {}".format(self.storage.camera_keys))

    def _filter_candidate_nodes(self):
        nodes_enough_viewed = set(
            node
            for node in self.storage.visibility_graph.nodes
            if len(self.storage.visibility_graph.nodes[node])
            >= self._min_n_frames_per_marker
        )
        try:
            nodes_connected_to_first_node = set(
                nx.node_connected_component(
                    self.storage.visibility_graph, self.storage.marker_keys[0]
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

    def _collect_data_for_optimization(self):
        camera_indices = []
        marker_indices = []
        markers_points_2d_detected = []
        for f_index, f_id in enumerate(self.storage.camera_keys):
            for n_index, n_id in enumerate(self.storage.marker_keys):
                if n_id in self.storage.keyframes[f_id]:
                    camera_indices.append(f_index)
                    marker_indices.append(n_index)
                    markers_points_2d_detected.append(
                        self.storage.keyframes[f_id][n_id]["verts"]
                    )

        camera_indices = np.array(camera_indices)
        marker_indices = np.array(marker_indices)
        markers_points_2d_detected = np.array(markers_points_2d_detected)

        try:
            markers_points_2d_detected = markers_points_2d_detected[:, :, 0, :]
        except IndexError:
            return

        camera_extrinsics_prv = {
            i: self.storage.camera_extrinsics_opt[k]
            for i, k in enumerate(self.storage.camera_keys)
            if k in self.storage.camera_extrinsics_opt
        }

        marker_extrinsics_prv = {
            i: self.storage.marker_extrinsics_opt[k]
            for i, k in enumerate(self.storage.marker_keys)
            if k in self.storage.marker_extrinsics_opt
        }

        data_for_optimization = DataForOptimization(
            camera_indices=camera_indices,
            marker_indices=marker_indices,
            markers_points_2d_detected=markers_points_2d_detected,
            camera_extrinsics_prv=camera_extrinsics_prv,
            marker_extrinsics_prv=marker_extrinsics_prv,
        )
        self.on_data_for_optimization_prepared(data_for_optimization)
