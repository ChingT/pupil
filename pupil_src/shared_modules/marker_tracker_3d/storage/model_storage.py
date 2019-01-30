import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import file_methods
from marker_tracker_3d import worker
from observable import Observable

logger = logging.getLogger(__name__)


class ModelStorage(Observable):
    def __init__(self, save_path):
        self._model_path = os.path.join(save_path, "marker_tracker_3d_model")

        self._visibility_graph_path = os.path.join(save_path, "visibility_graphs")

        self._set_to_default_values()

    def _set_to_default_values(self):
        self.visibility_graph = nx.MultiGraph()

        self.adding_observations = True
        self.current_frame_id = 0

        self.all_novel_markers = []
        self.n_new_novel_markers_added = 0
        self.model_being_updated = False

        # frame_id_to_extrinsics_opt: {frame id: optimized camera extrinsics (which is
        # composed of Rodrigues rotation vector and translation vector, which brings
        # points from the world coordinate system to the camera coordinate system)}
        self.frame_id_to_extrinsics_opt = {}

        # marker_id_to_extrinsics_opt: {marker id: optimized marker extrinsics}
        self.marker_id_to_extrinsics_opt = {}

        # marker_id_to_points_3d_opt: {marker id: 3d points of 4 vertices of the marker
        # in the world coordinate system}.
        # it is updated according to marker_id_to_extrinsics_opt by the function
        # extrinsics_to_marker_id_to_points_3d
        self.marker_id_to_points_3d_opt = {}

        # TODO: debug only; to be removed
        self.marker_id_to_points_3d_init = {}

        # TODO: redo origin_marker_id logic
        self.origin_marker_id = None

    def reset(self):
        self._set_to_default_values()

    def load_marker_tracker_3d_model_from_file(self):
        model = file_methods.Persistent_Dict(self._model_path)

        marker_id_to_extrinsics_opt = model.get("marker_id_to_extrinsics_opt", {})
        origin_marker_id = worker.utils.find_origin_marker_id(
            marker_id_to_extrinsics_opt
        )
        self.setup_origin_marker_id(origin_marker_id)

        for marker_id, extrinsics in marker_id_to_extrinsics_opt.items():
            self.marker_id_to_extrinsics_opt[marker_id] = np.array(extrinsics)
            self.marker_id_to_points_3d_opt[
                marker_id
            ] = worker.utils.convert_marker_extrinsics_to_points_3d(
                np.array(extrinsics)
            )

        if self.marker_id_to_extrinsics_opt:
            logger.info(
                "marker tracker 3d model with {0} markers has been loaded from "
                "{1}".format(len(marker_id_to_extrinsics_opt), self._model_path)
            )

    def export_marker_tracker_3d_model(self):
        marker_tracker_3d_model = file_methods.Persistent_Dict(self._model_path)
        marker_tracker_3d_model["marker_id_to_extrinsics_opt"] = {
            marker_id: extrinsics.tolist()
            for marker_id, extrinsics in self.marker_id_to_extrinsics_opt.items()
        }
        marker_tracker_3d_model.save()

        logger.info(
            "marker tracker 3d model with {0} markers has been exported to {1}".format(
                len(self.marker_id_to_extrinsics_opt), self._model_path
            )
        )

    def setup_origin_marker_id(self, origin_marker_id):
        if self.origin_marker_id is not None and origin_marker_id is not None:
            assert self.origin_marker_id == origin_marker_id
        self.origin_marker_id = origin_marker_id
        if origin_marker_id is not None:
            self.on_origin_marker_id_set()

    def on_origin_marker_id_set(self):
        pass

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
        else:
            self.marker_id_to_extrinsics_opt = {}
            self.marker_id_to_points_3d_opt = {}

    # TODO: debug only; to be removed
    def export_visibility_graph(self, show_unconnected_nodes=False):
        if not self.visibility_graph:
            return

        graph_vis = self.visibility_graph.copy()
        connected_component = nx.node_connected_component(
            self.visibility_graph, self.origin_marker_id
        )

        if not show_unconnected_nodes:
            if self.origin_marker_id not in self.visibility_graph:
                return
            unconnected_nodes = set(graph_vis.nodes) - set(connected_component)
            graph_vis.remove_nodes_from(unconnected_nodes)

        pos = nx.spring_layout(graph_vis, seed=0)

        nx.draw_networkx_nodes(
            graph_vis,
            pos,
            nodelist=list(graph_vis.nodes),
            node_color="g",
            node_size=100,
        )
        nx.draw_networkx_nodes(
            graph_vis,
            pos,
            nodelist=connected_component,
            node_color="orange",
            node_size=100,
        )
        nx.draw_networkx_nodes(
            graph_vis,
            pos,
            nodelist=list(
                set(graph_vis.nodes) & self.marker_id_to_extrinsics_opt.keys()
            ),
            node_color="r",
            node_size=100,
        )

        nx.draw_networkx_nodes(
            graph_vis,
            pos,
            nodelist=list(set(graph_vis.nodes) & {self.origin_marker_id}),
            node_color="brown",
            node_size=100,
        )
        nx.draw_networkx_edges(graph_vis, pos, width=1, alpha=0.1)
        nx.draw_networkx_labels(graph_vis, pos, font_size=7)

        plt.axis("off")
        save_name = os.path.join(
            self._visibility_graph_path,
            "frame-{0:03d}-{1}-{2}".format(
                self.current_frame_id,
                len(self.visibility_graph),
                len(self.marker_id_to_extrinsics_opt),
            ),
        )
        plt.savefig(save_name, dpi=300)
        plt.clf()

        logger.info("visibility_graph has been exported to {}".format(save_name))
