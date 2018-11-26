import collections
import itertools as it
import logging
import os

import networkx as nx
import numpy as np

from marker_tracker_3d import math
from marker_tracker_3d import utils
from marker_tracker_3d.camera_localizer import CameraLocalizer

logger = logging.getLogger(__name__)


class VisibilityGraphs:
    def __init__(
        self,
        camera_model,
        origin_marker_id=None,
        update_menu=None,
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

        self.camera_localizer = CameraLocalizer(camera_model)
        self.origin_marker_id = origin_marker_id
        self.update_menu = update_menu

        self.min_number_of_markers_per_frame = min_number_of_markers_per_frame
        self.min_number_of_frames_per_marker = min_number_of_frames_per_marker
        self.min_angle_diff = min_camera_angle_diff
        self.optimization_interval = optimization_interval
        self.select_keyframe_interval = select_keyframe_interval

        self.frame_id = 0
        self.count_frame = 0

        self.marker_keys = list()
        self.camera_keys = list()

        self.camera_extrinsics_opt = dict()
        self.marker_extrinsics_opt = collections.OrderedDict()

        self.keyframes = dict()
        self.visibility_graph_of_keyframes = nx.MultiGraph()
        self.visibility_graph_of_ready_markers = nx.MultiGraph()

    def reset(self):
        self.frame_id = 0
        self.count_frame = 0

        self.marker_keys = list()
        self.camera_keys = list()

        self.camera_extrinsics_opt = dict()
        self.marker_extrinsics_opt = collections.OrderedDict()

        self.keyframes = dict()
        self.visibility_graph_of_keyframes = nx.MultiGraph()
        self.visibility_graph_of_ready_markers = nx.MultiGraph()

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
        if self._decide_keyframe(candidate_marker_keys):
            self._add_keyframe(
                marker_detections, candidate_marker_keys, camera_extrinsics
            )
            self._add_to_graph(candidate_marker_keys, camera_extrinsics)
            self.frame_id += 1

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

        self.marker_keys = [origin_marker_id]
        self.marker_extrinsics_opt = {origin_marker_id: utils.marker_extrinsics_origin}
        self.update_menu()

    def _get_camera_extrinsics(self, marker_detections, camera_extrinsics):
        if camera_extrinsics is None:
            try:
                assert self.marker_extrinsics_opt
            except AssertionError:
                self._set_coordinate_system(marker_detections)

            camera_extrinsics = self.camera_localizer.get_camera_extrinsics(
                marker_detections, self.marker_extrinsics_opt
            )
        return camera_extrinsics

    def _get_candidate_marker_keys(self, marker_detections, camera_extrinsics):
        """
        get those markers in marker_detections, to which the rotation vector of the current camera pose is diverse enough
        """

        rvec, _ = utils.split_param(camera_extrinsics)

        candidate_marker_keys = list()
        for n_id in marker_detections.keys():
            if n_id in self.visibility_graph_of_keyframes.nodes and len(
                self.visibility_graph_of_keyframes.nodes[n_id]
            ):
                diff = math.closest_angle_diff(
                    rvec, list(self.visibility_graph_of_keyframes.nodes[n_id].values())
                )
                if diff > self.min_angle_diff:
                    candidate_marker_keys.append(n_id)
            else:
                candidate_marker_keys.append(n_id)

        return candidate_marker_keys

    def _decide_keyframe(self, candidate_marker_keys):
        """ decide if markers can be a keyframe """
        # TODO: come up a way to pick up keyframes without camera extrinsics

        if len(candidate_marker_keys) < self.min_number_of_markers_per_frame:
            return False

        logger.debug(
            "--> keyframe {0}; markers {1}".format(self.frame_id, candidate_marker_keys)
        )
        return True

    def _add_keyframe(
        self, marker_detections, candidate_marker_keys, camera_extrinsics
    ):
        self.keyframes[self.frame_id] = {
            k: marker_detections[k] for k in candidate_marker_keys
        }
        self.keyframes[self.frame_id]["previous_camera_extrinsics"] = camera_extrinsics

    def _add_to_graph(self, candidate_marker_keys, camera_extrinsics):
        """
        graph"s node: marker id; attributes: the keyframe id
        graph"s edge: keyframe id, where two markers shown in the same frame
        """

        # add frame_id as edges in the graph
        for u, v in list(it.combinations(candidate_marker_keys, 2)):
            self.visibility_graph_of_keyframes.add_edge(u, v, key=self.frame_id)

        # add frame_id as an attribute of the node
        rvec, _ = utils.split_param(camera_extrinsics)
        for n_id in candidate_marker_keys:
            self.visibility_graph_of_keyframes.nodes[n_id][self.frame_id] = rvec

    def get_data_for_optimization(self):
        # Do optimization when there are some new keyframes selected
        if self.frame_id % self.optimization_interval == 0:
            self._update_visibility_graph_of_ready_markers()
            self._update_camera_and_marker_keys()

            # prepare data for optimization
            data_for_optimization = self._prepare_data_for_optimization()
            return data_for_optimization

    def _update_visibility_graph_of_ready_markers(self):
        """
        find out ready markers for optimization
        """

        if self.marker_keys:
            self.visibility_graph_of_ready_markers = (
                self.visibility_graph_of_keyframes.copy()
            )
            while True:
                # remove the nodes which are not viewed more than self.min_number_of_frames_per_marker times
                nodes_not_candidate = [
                    n
                    for n in self.visibility_graph_of_ready_markers.nodes
                    if len(self.visibility_graph_of_ready_markers.nodes[n])
                    < self.min_number_of_frames_per_marker
                ]
                self._remove_nodes(nodes_not_candidate)

                if (
                    len(self.visibility_graph_of_ready_markers) == 0
                    or self.marker_keys[0] not in self.visibility_graph_of_ready_markers
                ):
                    return

                # remove the nodes which are not connected to the first node
                nodes_not_connected = list(
                    set(self.visibility_graph_of_ready_markers.nodes)
                    - set(
                        nx.node_connected_component(
                            self.visibility_graph_of_ready_markers, self.marker_keys[0]
                        )
                    )
                )
                self._remove_nodes(nodes_not_connected)

                if len(self.visibility_graph_of_ready_markers) == 0:
                    return

                if len(nodes_not_candidate) == 0 and len(nodes_not_connected) == 0:
                    return

    def _remove_nodes(self, nodes):
        """ remove nodes in the graph """

        self.visibility_graph_of_ready_markers.remove_nodes_from(nodes)

    def _update_camera_and_marker_keys(self):
        """ add new ids to self.marker_keys """

        if self.marker_keys:
            self.camera_keys = sorted(
                set(
                    f_id
                    for _, _, f_id in self.visibility_graph_of_ready_markers.edges(
                        keys=True
                    )
                )
            )
            logger.debug("self.camera_keys updated {}".format(self.camera_keys))

            self.marker_keys = [self.marker_keys[0]] + [
                n
                for n in self.visibility_graph_of_ready_markers.nodes
                if n != self.marker_keys[0]
            ]
            logger.debug("self.marker_keys updated {}".format(self.marker_keys))

    def _prepare_data_for_optimization(self):
        """ prepare data for optimization """

        camera_indices = list()
        marker_indices = list()
        markers_points_2d_detected = list()
        for f_id in self.camera_keys:
            for n_id in self.keyframes[f_id].keys() & set(self.marker_keys):
                camera_indices.append(self.camera_keys.index(f_id))
                marker_indices.append(self.marker_keys.index(n_id))
                markers_points_2d_detected.append(self.keyframes[f_id][n_id]["verts"])

        if len(markers_points_2d_detected):
            camera_indices = np.array(camera_indices)
            marker_indices = np.array(marker_indices)
            markers_points_2d_detected = np.array(markers_points_2d_detected)[
                :, :, 0, :
            ]
        else:
            return

        camera_extrinsics_prv = {}
        for i, k in enumerate(self.camera_keys):
            if k in self.camera_extrinsics_opt:
                camera_extrinsics_prv[i] = self.camera_extrinsics_opt[k]
            else:
                try:
                    camera_extrinsics_prv[i] = self.keyframes[k][
                        "previous_camera_extrinsics"
                    ].ravel()
                except KeyError:
                    raise KeyError("previous_camera_extrinsics should be in keyframes")

        marker_extrinsics_prv = {}
        for i, k in enumerate(self.marker_keys):
            if k in self.marker_extrinsics_opt:
                marker_extrinsics_prv[i] = self.marker_extrinsics_opt[k]

        data_for_optimization = (
            camera_indices,
            marker_indices,
            markers_points_2d_detected,
            camera_extrinsics_prv,
            marker_extrinsics_prv,
        )

        return data_for_optimization

    def get_updated_marker_extrinsics(self, optimization_result):
        """ process the results of optimization """

        try:
            camera_extrinsics_opt = optimization_result["camera_extrinsics_opt"]
            marker_extrinsics_opt = optimization_result["marker_extrinsics_opt"]
            camera_keys_failed = optimization_result["camera_keys_failed"]
            marker_keys_failed = optimization_result["marker_keys_failed"]
        except KeyError:
            return
        else:
            self._update_extrinsics(
                camera_extrinsics_opt,
                marker_extrinsics_opt,
                camera_keys_failed,
                marker_keys_failed,
            )

            self._discard_keyframes(camera_keys_failed)
            return self.marker_extrinsics_opt

    def _update_extrinsics(
        self,
        camera_extrinsics,
        marker_extrinsics,
        camera_keys_failed,
        marker_keys_failed,
    ):
        for i, p in enumerate(camera_extrinsics):
            if i not in camera_keys_failed:
                self.camera_extrinsics_opt[self.camera_keys[i]] = p
        for i, p in enumerate(marker_extrinsics):
            if i not in marker_keys_failed:
                self.marker_extrinsics_opt[self.marker_keys[i]] = p

        logger.info(
            "{} markers have been registered and updated".format(
                len(self.marker_extrinsics_opt)
            )
        )

    def _discard_keyframes(self, camera_keys_failed):
        """ if the optimization failed, remove those frame_id, which make optimization fail from self.keyframes
        update keyframes, the graph """

        # TODO: check the image

        if not camera_keys_failed:
            return
        failed_keyframes = set(self.camera_keys[i] for i in camera_keys_failed)

        self._del_failed_keyframes(failed_keyframes)
        self._remove_failed_frame_id_from_graph(failed_keyframes)
        self._remove_failed_marker_keys()

    def _del_failed_keyframes(self, failed_keyframes):
        for f_id in failed_keyframes:
            try:
                del self.keyframes[f_id]
            except KeyError:
                logger.debug("{} is not in keyframes".format(f_id))
        logger.debug("remove from keyframes: {}".format(failed_keyframes))

    def _remove_failed_frame_id_from_graph(self, failed_keyframes):
        redundant_edges = [
            (n_id, neighbor, f_id)
            for n_id, neighbor, f_id in self.visibility_graph_of_keyframes.edges(
                keys=True
            )
            if f_id in failed_keyframes
        ]

        self.visibility_graph_of_keyframes.remove_edges_from(redundant_edges)

        for f_id in failed_keyframes:
            for n_id in set(n for n, _, f in redundant_edges if f == f_id) | set(
                n for _, n, f in redundant_edges if f == f_id
            ):
                del self.visibility_graph_of_keyframes.nodes[n_id][f_id]

    def _remove_failed_marker_keys(self):
        fail_marker_keys = set(self.marker_keys) - set(
            self.marker_extrinsics_opt.keys()
        )
        for k in fail_marker_keys:
            self.marker_keys.remove(k)
        logger.debug("remove from marker_keys: {}".format(fail_marker_keys))

    # For debug
    def vis_graph(self, save_path):
        import matplotlib.pyplot as plt

        if len(self.visibility_graph_of_keyframes) and self.marker_keys:
            graph_vis = self.visibility_graph_of_keyframes.copy()
            all_nodes = list(graph_vis.nodes)

            pos = nx.spring_layout(graph_vis, seed=0)  # positions for all nodes
            pos_label = dict((n, pos[n] + 0.05) for n in pos)

            nx.draw_networkx_nodes(
                graph_vis, pos, nodelist=all_nodes, node_color="g", node_size=100
            )
            if self.marker_keys[0] in self.visibility_graph_of_ready_markers:
                connected_component = nx.node_connected_component(
                    self.visibility_graph_of_ready_markers, self.marker_keys[0]
                )
                nx.draw_networkx_nodes(
                    graph_vis,
                    pos,
                    nodelist=connected_component,
                    node_color="r",
                    node_size=100,
                )
            nx.draw_networkx_edges(graph_vis, pos, width=1, alpha=0.1)
            nx.draw_networkx_labels(graph_vis, pos, font_size=7)

            labels = dict(
                (n, self.marker_keys.index(n) if n in self.marker_keys else None)
                for n in graph_vis.nodes()
            )
            nx.draw_networkx_labels(
                graph_vis, pos=pos_label, labels=labels, font_size=6, font_color="b"
            )

            plt.axis("off")
            save_name = os.path.join(
                save_path,
                "weighted_graph-{0:03d}-{1}-{2}-{3}.png".format(
                    self.frame_id,
                    len(self.visibility_graph_of_keyframes),
                    len(self.visibility_graph_of_ready_markers),
                    len(self.marker_extrinsics_opt),
                ),
            )
            plt.savefig(save_name)
            plt.clf()
