import itertools as it
import logging

import networkx as nx
import numpy as np

from marker_tracker_3d import math
from marker_tracker_3d import utils
from marker_tracker_3d.camera_localizer import CameraLocalizer
from observable import Observable

logger = logging.getLogger(__name__)


class VisibilityGraphs(Observable):
    def __init__(
        self,
        storage,
        camera_model,
        origin_marker_id=None,
        min_number_of_markers_per_frame=3,
        min_number_of_frames_per_marker=2,
        min_camera_angle_diff=0.1,
        optimization_interval=2,
        select_keyframe_interval=6,
    ):
        assert min_number_of_markers_per_frame >= 2
        assert min_number_of_frames_per_marker >= 2
        assert min_camera_angle_diff > 0
        assert optimization_interval >= 1
        assert select_keyframe_interval >= 1

        self.storage = storage
        self.camera_localizer = CameraLocalizer(camera_model)
        self.origin_marker_id = origin_marker_id

        self.min_number_of_markers_per_frame = min_number_of_markers_per_frame
        self.min_number_of_frames_per_marker = min_number_of_frames_per_marker
        self.min_angle_diff = min_camera_angle_diff
        self.optimization_interval = optimization_interval
        self.select_keyframe_interval = select_keyframe_interval

        self.frame_id = -1
        self.frame_id_last_opt = self.frame_id
        self.count_frame = 0

    def reset(self):
        self.frame_id = -1
        self.frame_id_last_opt = self.frame_id
        self.count_frame = 0
        self.on_update_menu()

    def _add_observer_to_keyframe_added(self):
        self.add_observer("on_keyframe_added", self.get_data_for_optimization)

    def _remove_observer_from_keyframe_added(self):
        self.remove_observer("on_keyframe_added", self.get_data_for_optimization)

    def add_marker_detections(self, marker_detections, camera_extrinsics):
        self.count_frame += 1
        if self.count_frame >= self.select_keyframe_interval:
            self.count_frame = 0

            self._add_markers_to_visibility_graph_of_keyframes(
                marker_detections, camera_extrinsics
            )

    def _add_markers_to_visibility_graph_of_keyframes(
        self, marker_detections, camera_extrinsics
    ):
        """ pick up keyframe and update visibility graph of keyframes """

        camera_extrinsics = self._get_camera_extrinsics(
            marker_detections, camera_extrinsics
        )
        if camera_extrinsics is None:
            return

        candidate_marker_keys = self._get_candidate_marker_keys(
            marker_detections, camera_extrinsics
        )
        if len(candidate_marker_keys) >= self.min_number_of_markers_per_frame:
            self.frame_id += 1
            self._add_keyframe(
                marker_detections, candidate_marker_keys, camera_extrinsics
            )
            self._add_to_graph(candidate_marker_keys, camera_extrinsics)
            self.on_keyframe_added()

    def on_keyframe_added(self):
        pass

    def _get_camera_extrinsics(self, marker_detections, camera_extrinsics):
        if camera_extrinsics is None:
            if not self.storage.marker_extrinsics_opt:
                self._set_coordinate_system(marker_detections)

            camera_extrinsics = self.camera_localizer.get_camera_extrinsics(
                marker_detections, self.storage.marker_extrinsics_opt
            )
        return camera_extrinsics

    def _set_coordinate_system(self, marker_detections):
        if not marker_detections:
            return

        if self.origin_marker_id:
            if self.origin_marker_id in marker_detections:
                origin_marker_id = self.origin_marker_id
            else:
                return
        else:
            origin_marker_id = list(marker_detections.keys())[0]

        self.storage.marker_keys = [origin_marker_id]
        self.storage.marker_extrinsics_opt = {
            origin_marker_id: utils.marker_extrinsics_origin
        }
        self.storage.marker_points_3d_opt = {origin_marker_id: utils.marker_df}
        self.on_update_menu()

    def on_update_menu(self):
        pass

    def _get_candidate_marker_keys(self, marker_detections, camera_extrinsics):
        """
        get those markers in marker_detections,
        to which the rotation vector of the current camera pose is diverse enough
        """
        # TODO: come up a way to pick up keyframes without camera extrinsics

        rvec, _ = utils.split_param(camera_extrinsics)

        candidate_marker_keys = []
        for n_id in marker_detections.keys():
            try:
                rvecs_saved = self.storage.visibility_graph_of_keyframes.nodes[
                    n_id
                ].values()
            except KeyError:
                candidate_marker_keys.append(n_id)
                continue

            diff = math.closest_angle_diff(rvec, list(rvecs_saved))
            if diff > self.min_angle_diff:
                candidate_marker_keys.append(n_id)

        return candidate_marker_keys

    def _add_keyframe(
        self, marker_detections, candidate_marker_keys, camera_extrinsics
    ):
        self.storage.keyframes[self.frame_id] = {
            k: marker_detections[k] for k in candidate_marker_keys
        }
        self.storage.keyframes[self.frame_id][
            "previous_camera_extrinsics"
        ] = camera_extrinsics

        logger.debug(
            "--> keyframe {0}; markers {1}".format(self.frame_id, candidate_marker_keys)
        )

    def _add_to_graph(self, candidate_marker_keys, camera_extrinsics):
        """
        graph"s node: marker id; attributes: the keyframe id
        graph"s edge: keyframe id, where two markers shown in the same frame
        """

        # add frame_id as edges in the graph
        for u, v in list(it.combinations(candidate_marker_keys, 2)):
            self.storage.visibility_graph_of_keyframes.add_edge(u, v, key=self.frame_id)

        # add frame_id as an attribute of the node
        rvec, _ = utils.split_param(camera_extrinsics)
        for n_id in candidate_marker_keys:
            self.storage.visibility_graph_of_keyframes.nodes[n_id][self.frame_id] = rvec

    def get_data_for_optimization(self):
        # Do optimization when there are some new keyframes selected
        if self.frame_id - self.frame_id_last_opt >= self.optimization_interval:
            self.storage.visibility_graph_of_ready_markers = (
                self._get_visibility_graph_of_ready_markers()
            )
            self._update_camera_and_marker_keys()

            self._prepare_data_for_optimization()

    def _get_visibility_graph_of_ready_markers(self):
        """ find out ready markers for optimization """

        visibility_graph_of_ready_markers = (
            self.storage.visibility_graph_of_keyframes.copy()
        )

        while True:
            nodes_less_viewed = self._find_nodes_less_viewed(
                visibility_graph_of_ready_markers
            )
            visibility_graph_of_ready_markers.remove_nodes_from(nodes_less_viewed)

            nodes_not_connected = self._find_nodes_not_connected_to_first_node(
                visibility_graph_of_ready_markers
            )
            visibility_graph_of_ready_markers.remove_nodes_from(nodes_not_connected)

            if not nodes_less_viewed and not nodes_not_connected:
                break

        return visibility_graph_of_ready_markers

    def _find_nodes_less_viewed(self, visibility_graph_of_ready_markers):
        """ find the nodes which are not viewed more than self.min_number_of_frames_per_marker times"""

        nodes_less_viewed = set(
            n
            for n in visibility_graph_of_ready_markers.nodes
            if len(visibility_graph_of_ready_markers.nodes[n])
            < self.min_number_of_frames_per_marker
        )
        return nodes_less_viewed

    def _find_nodes_not_connected_to_first_node(
        self, visibility_graph_of_ready_markers
    ):
        """ find the nodes which are not connected to the first node """

        try:
            nodes_connected_to_first_node = nx.node_connected_component(
                visibility_graph_of_ready_markers, self.storage.marker_keys[0]
            )
        except KeyError:
            nodes_connected_to_first_node = set()
        except IndexError:
            nodes_connected_to_first_node = set()

        nodes_not_connected_to_first_node = (
            visibility_graph_of_ready_markers.nodes - nodes_connected_to_first_node
        )
        return nodes_not_connected_to_first_node

    def _update_camera_and_marker_keys(self):
        try:
            self.storage.marker_keys = [self.storage.marker_keys[0]]
        except IndexError:
            return

        self.storage.marker_keys += [
            n
            for n in self.storage.visibility_graph_of_ready_markers.nodes
            if n != self.storage.marker_keys[0]
        ]
        logger.debug("marker_keys updated {}".format(self.storage.marker_keys))

        self.storage.camera_keys = sorted(
            set(
                f_id
                for _, _, f_id in self.storage.visibility_graph_of_ready_markers.edges(
                    keys=True
                )
            )
        )
        logger.debug("camera_keys updated {}".format(self.storage.camera_keys))

    def _prepare_data_for_optimization(self):
        """ prepare data for optimization """

        camera_indices = []
        marker_indices = []
        markers_points_2d_detected = []
        for f_id in self.storage.camera_keys:
            for n_id in self.storage.keyframes[f_id].keys() & set(
                self.storage.marker_keys
            ):
                camera_indices.append(self.storage.camera_keys.index(f_id))
                marker_indices.append(self.storage.marker_keys.index(n_id))
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
            if k in self.storage.camera_extrinsics_opt
            else self.storage.keyframes[k]["previous_camera_extrinsics"]
            for i, k in enumerate(self.storage.camera_keys)
        }

        marker_extrinsics_prv = {
            i: self.storage.marker_extrinsics_opt[k]
            for i, k in enumerate(self.storage.marker_keys)
            if k in self.storage.marker_extrinsics_opt
        }

        data_for_optimization = utils.DataForOptimization(
            camera_indices,
            marker_indices,
            markers_points_2d_detected,
            camera_extrinsics_prv,
            marker_extrinsics_prv,
        )
        self.on_data_for_optimization_prepared(data_for_optimization)

    def on_data_for_optimization_prepared(self, data_for_optimization):
        pass

    def get_updated_marker_extrinsics(self, optimization_result):
        """ process the results of optimization """

        try:
            camera_extrinsics_opt = optimization_result.camera_extrinsics_opt
            marker_extrinsics_opt = optimization_result.marker_extrinsics_opt
            camera_keys_failed = optimization_result.camera_keys_failed
            marker_keys_failed = optimization_result.marker_keys_failed
        except AttributeError:
            return

        self._update_extrinsics(
            camera_extrinsics_opt,
            marker_extrinsics_opt,
            camera_keys_failed,
            marker_keys_failed,
        )

        self._discard_keyframes(camera_keys_failed)

    def _update_extrinsics(
        self,
        camera_extrinsics,
        marker_extrinsics,
        camera_keys_failed,
        marker_keys_failed,
    ):
        for i, p in enumerate(camera_extrinsics):
            if i not in camera_keys_failed:
                self.storage.camera_extrinsics_opt[self.storage.camera_keys[i]] = p
        for i, p in enumerate(marker_extrinsics):
            if i not in marker_keys_failed:
                self.storage.marker_extrinsics_opt[self.storage.marker_keys[i]] = p
                self.storage.marker_points_3d_opt[
                    self.storage.marker_keys[i]
                ] = utils.params_to_points_3d(p)[0]

        logger.info(
            "{} markers have been registered and updated".format(
                len(self.storage.marker_extrinsics_opt)
            )
        )

    def _discard_keyframes(self, camera_keys_failed):
        """ if the optimization failed, remove those frame_id, which make optimization fail
        from self.storage.keyframes, update keyframes and the graph """

        # TODO: check the image

        if not camera_keys_failed:
            return
        failed_keyframes = set(self.storage.camera_keys[i] for i in camera_keys_failed)

        self._del_failed_keyframes(failed_keyframes)
        self._remove_failed_frame_id_from_graph(failed_keyframes)
        self._remove_failed_marker_keys()

    def _del_failed_keyframes(self, failed_keyframes):
        for f_id in failed_keyframes:
            try:
                del self.storage.keyframes[f_id]
            except KeyError:
                logger.debug("{} is not in keyframes".format(f_id))
        logger.debug("remove from keyframes: {}".format(failed_keyframes))

    def _remove_failed_frame_id_from_graph(self, failed_keyframes):
        redundant_edges = [
            (n_id, neighbor, f_id)
            for n_id, neighbor, f_id in self.storage.visibility_graph_of_keyframes.edges(
                keys=True
            )
            if f_id in failed_keyframes
        ]
        self.storage.visibility_graph_of_keyframes.remove_edges_from(redundant_edges)

        for f_id in failed_keyframes:
            for n_id in set(n for n, _, f in redundant_edges if f == f_id) | set(
                n for _, n, f in redundant_edges if f == f_id
            ):
                del self.storage.visibility_graph_of_keyframes.nodes[n_id][f_id]

        redundant_nodes = [
            n_id
            for n_id in self.storage.visibility_graph_of_keyframes.nodes
            if not self.storage.visibility_graph_of_keyframes.nodes[n_id]
        ]
        self.storage.visibility_graph_of_keyframes.remove_nodes_from(redundant_nodes)

    def _remove_failed_marker_keys(self):
        fail_marker_keys = set(self.storage.marker_keys) - set(
            self.storage.marker_extrinsics_opt.keys()
        )
        for k in fail_marker_keys:
            self.storage.marker_keys.remove(k)
        logger.debug("remove from marker_keys: {}".format(fail_marker_keys))
