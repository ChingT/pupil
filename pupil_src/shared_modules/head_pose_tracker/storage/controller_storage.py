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
import itertools as it
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx

import file_methods
from head_pose_tracker import worker

logger = logging.getLogger(__name__)


KeyMarker = collections.namedtuple(
    "KeyMarker", ["frame_id", "marker_id", "verts", "bin"]
)


class ControllerStorage:
    def __init__(self, save_path):
        self._all_camera_extrinsics_path = os.path.join(
            save_path, "offline_data", "all_camera_extrinsics"
        )
        self._all_marker_id_to_detections_path = os.path.join(
            save_path, "offline_data", "all_marker_id_to_detections"
        )
        self._all_key_markers_path = os.path.join(
            save_path, "offline_data", "all_key_markers"
        )

        self._all_key_edges_path = os.path.join(
            save_path, "offline_data", "all_key_edges"
        )
        self._visibility_graph_path = os.path.join(save_path, "visibility_graph")

        self._set_to_default_values()

        # for cache
        self.all_marker_id_to_detections = {}

        self.load_all_camera_extrinsics()
        self.load_all_marker_id_to_detections()

        self.progress = 0

    def _set_to_default_values(self):
        self._not_localized_count = 0

        self.visibility_graph = nx.MultiGraph()

        # for export
        self.all_camera_extrinsics = {}

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
        self.marker_id_to_detections = marker_id_to_detections

    def save_all_marker_id_to_detections(
        self, marker_id_to_detections, current_frame_id
    ):
        self.all_marker_id_to_detections[current_frame_id] = marker_id_to_detections

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, _progress):
        self._progress = _progress

        if _progress == 100:
            self.status = "Successfully completed"
        else:
            self.status = "{:.0f}% complete".format(self.progress * 100)

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

    def save_all_camera_extrinsics(self, camera_extrinsics, current_frame_id):
        try:
            self.all_camera_extrinsics[current_frame_id] = camera_extrinsics.tolist()
        except AttributeError:
            self.all_camera_extrinsics[current_frame_id] = None

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

    def save_key_markers(self, marker_id_to_detections, current_frame_id):
        key_markers = [
            KeyMarker(current_frame_id, marker_id, detection["verts"], detection["bin"])
            for marker_id, detection in marker_id_to_detections.items()
        ]
        self.all_key_markers += key_markers

        marker_ids = [marker.marker_id for marker in key_markers]
        key_edges = [
            (marker_id1, marker_id2, current_frame_id)
            for marker_id1, marker_id2 in list(it.combinations(marker_ids, 2))
        ]

        self.visibility_graph.add_edges_from(key_edges)

        self.all_key_edges += key_edges

    def export_all_marker_id_to_detections(self):
        if not self.all_marker_id_to_detections:
            return

        file_methods.save_object(
            self.all_marker_id_to_detections, self._all_marker_id_to_detections_path
        )

        logger.info(
            "all_marker_id_to_detections from {0} frames has been exported to "
            "{1}".format(
                len(self.all_marker_id_to_detections),
                self._all_marker_id_to_detections_path,
            )
        )

    def load_all_marker_id_to_detections(self):
        try:
            all_marker_id_to_detections = file_methods.load_object(
                self._all_marker_id_to_detections_path
            )
        except FileNotFoundError:
            return

        self.all_marker_id_to_detections = all_marker_id_to_detections

        logger.info(
            "all_marker_id_to_detections from {0} frames has been loaded from "
            "{1}".format(
                len(all_marker_id_to_detections), self._all_marker_id_to_detections_path
            )
        )

    def export_all_camera_extrinsics(self):
        if not self.all_camera_extrinsics:
            return

        file_methods.save_object(
            self.all_camera_extrinsics, self._all_camera_extrinsics_path
        )

        logger.info(
            "camera extrinsics from {0} frames has been exported to {1}".format(
                len(self.all_camera_extrinsics), self._all_camera_extrinsics_path
            )
        )

    def load_all_camera_extrinsics(self):
        try:
            all_camera_extrinsics = file_methods.load_object(
                self._all_camera_extrinsics_path
            )
        except FileNotFoundError:
            return

        self.all_camera_extrinsics = all_camera_extrinsics

        logger.info(
            "all_camera_extrinsics from {0} frames has been loaded from {1}".format(
                len(all_camera_extrinsics), self._all_camera_extrinsics_path
            )
        )

    def export_all_key_markers(self):
        if not self.all_key_markers:
            return

        file_methods.save_object(self.all_key_markers, self._all_key_markers_path)

        logger.info(
            "{0} all_key_markers has been exported to {1}".format(
                len(self.all_key_markers), self._all_key_markers_path
            )
        )

    def load_all_key_markers(self):
        try:
            all_key_markers = file_methods.load_object(self._all_key_markers_path)
        except FileNotFoundError:
            return

        self.all_key_markers = [
            KeyMarker(*key_marker) for key_marker in all_key_markers
        ]

        logger.info(
            "{0} all_key_markers has been loaded from {1}".format(
                len(self.all_key_markers), self._all_key_markers_path
            )
        )

    def export_all_key_edges(self):
        if not self.all_key_edges:
            return

        file_methods.save_object(self.all_key_edges, self._all_key_edges_path)

        logger.info(
            "{0} all_key_edges has been exported to {1}".format(
                len(self.all_key_edges), self._all_key_edges_path
            )
        )

    def load_all_key_edges(self):
        try:
            all_key_edges = file_methods.load_object(self._all_key_edges_path)
        except FileNotFoundError:
            return

        self.all_key_edges = all_key_edges
        self.visibility_graph.add_edges_from(all_key_edges)

        logger.info(
            "{0} all_key_edges has been loaded from {1}".format(
                len(self.all_key_edges), self._all_key_edges_path
            )
        )

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
