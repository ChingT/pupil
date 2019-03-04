"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import file_methods
from head_pose_tracker import worker
from observable import Observable

logger = logging.getLogger(__name__)


class ModelStorage(Observable):
    def __init__(self, predetermined_origin_marker_id, save_path):
        self._predetermined_origin_marker_id = predetermined_origin_marker_id

        self._model_path = os.path.join(save_path, "markers_3d_model")
        self._visibility_graph_path = os.path.join(save_path, "visibility_graph")

        self.optimize_3d_model = False
        self.optimize_camera_intrinsics = False

        self._set_to_default_values()

        self.load_markers_3d_model_from_file()

    def _set_to_default_values(self):
        self.visibility_graph = nx.MultiGraph()
        self.origin_marker_id = self._predetermined_origin_marker_id

        # {frame id: optimized camera extrinsics (which is composed of Rodrigues
        # rotation vector and translation vector, which brings points from the world
        # coordinate system to the camera coordinate system)}
        self.frame_id_to_extrinsics_opt = {}

        # {marker id: optimized marker extrinsics}
        self.marker_id_to_extrinsics_opt = {}

        # {marker id: 3d points of 4 vertices of the marker in the world coordinate
        # system}. It is updated according to marker_id_to_extrinsics_opt by the
        # function extrinsics_to_marker_id_to_points_3d
        self.marker_id_to_points_3d_opt = {}

        # TODO: debug only; to be removed
        self.marker_id_to_points_3d_init = {}

        self.calculate_points_3d_centroid()

    def reset(self):
        self._set_to_default_values()

    def calculate_points_3d_centroid(self):
        marker_id_to_points_3d = [
            points_3d for points_3d in self.marker_id_to_points_3d_opt.values()
        ]
        try:
            self.points_3d_centroid = np.mean(marker_id_to_points_3d, axis=(0, 1))
        except IndexError:
            self.points_3d_centroid = np.zeros((3,), dtype=np.float32)

    def load_markers_3d_model_from_file(self):
        try:
            marker_id_to_extrinsics_opt = file_methods.load_object(self._model_path)
        except FileNotFoundError:
            return

        marker_id_to_extrinsics_opt = {
            marker_id: np.array(extrinsics)
            for marker_id, extrinsics in marker_id_to_extrinsics_opt.items()
        }

        origin_marker_id = worker.utils.find_origin_marker_id(
            marker_id_to_extrinsics_opt
        )

        if self.origin_marker_id is None or self.origin_marker_id == origin_marker_id:
            marker_id_to_extrinsics_opt_converted = marker_id_to_extrinsics_opt
            self.origin_marker_id = origin_marker_id
        else:
            marker_id_to_extrinsics_opt_converted = self._convert_coordinate_system(
                marker_id_to_extrinsics_opt, self.marker_id_to_extrinsics_opt
            )
            if marker_id_to_extrinsics_opt_converted is None:
                logger.warning("cannot load marker tracker 3d model from file")
                return

        for marker_id, extrinsics in marker_id_to_extrinsics_opt_converted.items():
            self.marker_id_to_extrinsics_opt[marker_id] = extrinsics
            self.marker_id_to_points_3d_opt[
                marker_id
            ] = worker.utils.convert_marker_extrinsics_to_points_3d(extrinsics)

        self.calculate_points_3d_centroid()

        logger.info(
            "marker tracker 3d model with {0} markers has been loaded from "
            "{1}".format(len(marker_id_to_extrinsics_opt), self._model_path)
        )

    def _convert_coordinate_system(
        self, marker_id_to_extrinsics_opt_old, marker_id_to_extrinsics_opt_new
    ):
        try:
            common_key = list(
                set(marker_id_to_extrinsics_opt_old.keys())
                & set(marker_id_to_extrinsics_opt_new.keys())
            )[0]
        except IndexError:
            return None

        extrinsic_matrix_old = worker.utils.convert_extrinsic_to_matrix(
            marker_id_to_extrinsics_opt_old[common_key]
        )
        extrinsic_matrix_new = worker.utils.convert_extrinsic_to_matrix(
            marker_id_to_extrinsics_opt_new[common_key]
        )
        transform_matrix = np.matmul(
            extrinsic_matrix_new, np.linalg.inv(extrinsic_matrix_old)
        )

        new = {
            marker_id: worker.utils.convert_matrix_to_extrinsic(
                np.matmul(
                    transform_matrix,
                    worker.utils.convert_extrinsic_to_matrix(extrinsics),
                )
            )
            for marker_id, extrinsics in marker_id_to_extrinsics_opt_old.items()
        }
        if self.origin_marker_id not in new or np.allclose(
            new[self.origin_marker_id], worker.utils.get_marker_extrinsics_origin()
        ):
            return new
        else:
            return None

    def export_markers_3d_model_to_file(self):
        if self.marker_id_to_extrinsics_opt:
            marker_id_to_extrinsics_opt = {
                marker_id: extrinsics.tolist()
                for marker_id, extrinsics in self.marker_id_to_extrinsics_opt.items()
            }
            file_methods.save_object(marker_id_to_extrinsics_opt, self._model_path)

            logger.info(
                "marker tracker 3d model with {0} markers has been exported to {1}".format(
                    len(marker_id_to_extrinsics_opt), self._model_path
                )
            )
        else:
            logger.info("marker tracker 3d model has not yet built up")

    @property
    def origin_marker_id(self):
        return self._origin_marker_id

    @origin_marker_id.setter
    def origin_marker_id(self, origin_marker_id):
        self._origin_marker_id = origin_marker_id
        if origin_marker_id is not None:
            self.marker_id_to_extrinsics_opt = {
                origin_marker_id: worker.utils.get_marker_extrinsics_origin()
            }
            self.marker_id_to_points_3d_opt = {
                origin_marker_id: worker.utils.get_marker_points_3d_origin()
            }
            self.visibility_graph.add_node(origin_marker_id)

            logger.info(
                "The marker with id {} is defined as the origin of the coordinate "
                "system".format(origin_marker_id)
            )
            self.on_origin_marker_id_set()
        else:
            self.marker_id_to_extrinsics_opt = {}
            self.marker_id_to_points_3d_opt = {}

    def on_origin_marker_id_set(self):
        pass

    # TODO: debug only; to be removed
    def export_visibility_graph(self, show_unconnected_nodes=True):
        if not self.visibility_graph:
            return

        graph_vis = self.visibility_graph.copy()

        try:
            connected_component = nx.node_connected_component(
                self.visibility_graph, self.origin_marker_id
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
        draw_nodes(
            list(set(all_nodes) & self.marker_id_to_points_3d_init.keys()), "blue"
        )
        draw_nodes(
            list(set(all_nodes) & self.marker_id_to_extrinsics_opt.keys()), "red"
        )
        draw_nodes([self.origin_marker_id], "brown")

        nx.draw_networkx_edges(graph_vis, pos, width=1, alpha=0.1)
        nx.draw_networkx_labels(graph_vis, pos, font_size=7)

        plt.axis("off")
        save_name = os.path.join(
            self._visibility_graph_path,
            "frame-{0}-{1}.png".format(
                len(self.visibility_graph), len(self.marker_id_to_extrinsics_opt)
            ),
        )
        try:
            plt.savefig(save_name, dpi=300)
        except FileNotFoundError:
            os.makedirs(self._visibility_graph_path)
            plt.savefig(save_name, dpi=300)

        plt.clf()

        logger.info("visibility_graph has been exported to {}".format(save_name))
