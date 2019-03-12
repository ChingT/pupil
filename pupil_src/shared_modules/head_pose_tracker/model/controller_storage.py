"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx

from head_pose_tracker import worker

logger = logging.getLogger(__name__)


KeyMarker = collections.namedtuple(
    "KeyMarker", ["frame_id", "marker_id", "verts", "bin"]
)


class ControllerStorage:
    def __init__(self, save_path):
        self._visibility_graph_path = os.path.join(save_path, "visibility_graph")

        self._set_to_default_values()

    def _set_to_default_values(self):
        self._not_localized_count = 0

        self.visibility_graph = nx.MultiGraph()

        # for drawing in 2d window
        self.marker_id_to_detections = {}

        # for drawing in 3d window
        self.recent_camera_traces = collections.deque(maxlen=300)
        self.camera_pose_matrix = None
        self._camera_extrinsics = None
        self.camera_extrinsics = None

        self.all_key_markers = []
        self.all_key_edges = []
        self.n_key_markers_processed = 0

    def reset(self):
        self._set_to_default_values()

    def update_current_marker_id_to_detections(self, marker_id_to_detections):
        if marker_id_to_detections is None:
            self.marker_id_to_detections = {}
        else:
            self.marker_id_to_detections = marker_id_to_detections

    @property
    def marker_id_to_detections(self):
        return self._marker_id_to_detections

    @marker_id_to_detections.setter
    def marker_id_to_detections(self, _marker_id_to_detections):
        self._marker_id_to_detections = _marker_id_to_detections

    def update_current_camera_extrinsics(self, camera_extrinsics):
        self.camera_extrinsics = camera_extrinsics

    def update_current_camera_pose(self, camera_extrinsics):
        camera_poses = worker.utils.get_camera_pose(camera_extrinsics)
        self.recent_camera_traces.append(camera_poses[3:6])
        self.camera_pose_matrix = worker.utils.convert_extrinsic_to_matrix(camera_poses)

    @property
    def camera_extrinsics(self):
        return self._camera_extrinsics

    @camera_extrinsics.setter
    def camera_extrinsics(self, _camera_extrinsics):
        if _camera_extrinsics is not None:
            self._camera_extrinsics = _camera_extrinsics
            self._not_localized_count = 0
        else:
            # Do not set camera_extrinsics to None to ensure
            # a decent initial guess for the next solvePnP call;
            # except when there are multiple frames which could not be localized,
            # then set camera_extrinsics to None
            if self._not_localized_count >= 3:
                self._camera_extrinsics = None
            self._not_localized_count += 1

    # TODO: debug only; to be removed
    def export_visibility_graph(
        self,
        origin_marker_id,
        marker_id_to_extrinsics_opt_keys,
        marker_id_to_extrinsics_init_keys,
        show_unconnected_nodes=True,
    ):
        if not self.visibility_graph:
            return

        graph_vis = self.visibility_graph.copy()

        try:
            connected_component = nx.node_connected_component(
                self.visibility_graph, origin_marker_id
            )
        except KeyError:
            return

        if not show_unconnected_nodes:
            unconnected_nodes = set(graph_vis.nodes) - set(connected_component)
            graph_vis.remove_nodes_from(unconnected_nodes)

        pos = nx.spring_layout(graph_vis, seed=0)

        def draw_nodes(nodelist, node_color):
            nx.draw_networkx_nodes(
                graph_vis, pos, nodelist=nodelist, node_color=node_color, node_size=100
            )

        all_nodes = list(graph_vis.nodes)
        draw_nodes(all_nodes, "green")
        draw_nodes(list(set(all_nodes) & marker_id_to_extrinsics_init_keys), "blue")
        draw_nodes(list(set(all_nodes) & marker_id_to_extrinsics_opt_keys), "red")
        draw_nodes([origin_marker_id], "brown")

        nx.draw_networkx_edges(graph_vis, pos, width=1, alpha=0.1)
        nx.draw_networkx_labels(graph_vis, pos, font_size=7)

        plt.axis("off")
        save_name = os.path.join(
            self._visibility_graph_path,
            "frame-{0}-{1}.png".format(
                len(self.visibility_graph), len(marker_id_to_extrinsics_opt_keys)
            ),
        )
        try:
            plt.savefig(save_name, dpi=300)
        except FileNotFoundError:
            os.makedirs(self._visibility_graph_path)
            plt.savefig(save_name, dpi=300)

        plt.clf()

        logger.info("visibility_graph has been exported to {}".format(save_name))
