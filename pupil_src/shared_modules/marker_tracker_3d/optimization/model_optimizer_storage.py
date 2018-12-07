import datetime
import os

import networkx as nx

from marker_tracker_3d import utils


class ModelOptimizerStorage:
    def __init__(self):
        self.frame_id = 0

        self.marker_keys = []
        self.camera_keys = []
        self.keyframes = {}
        self.camera_extrinsics_opt = {}  # contain camera_extrinsics every frame
        self.marker_extrinsics_opt = {}
        self.marker_points_3d_opt = {}
        self.visibility_graph = nx.MultiGraph()

        # for export_data
        root = os.path.join(os.path.split(__file__)[0], "storage")
        now = datetime.datetime.now()
        now_str = "%02d%02d%02d-%02d%02d" % (
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
        )
        self.save_path = os.path.join(root, now_str)

    def export_data(self):
        dicts = {
            "marker_keys": self.marker_keys,
            "camera_keys": self.camera_keys,
            "keyframes": self.keyframes,
        }

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        utils.save_params_dicts(save_path=self.save_path, dicts=dicts)
        self.save_graph()

    def reset(self):
        self.marker_keys = []
        self.camera_keys = []
        self.keyframes = {}
        self.camera_extrinsics_opt = {}
        self.marker_extrinsics_opt = {}
        self.marker_points_3d_opt = {}
        self.visibility_graph = nx.MultiGraph()

    # For debug
    def save_graph(self):
        import matplotlib.pyplot as plt

        if self.visibility_graph and self.marker_keys:
            graph_vis = self.visibility_graph.copy()
            all_nodes = list(graph_vis.nodes)

            pos = nx.spring_layout(graph_vis, seed=0)  # positions for all nodes
            pos_label = dict((n, pos[n] + 0.05) for n in pos)

            nx.draw_networkx_nodes(
                graph_vis, pos, nodelist=all_nodes, node_color="g", node_size=100
            )
            nx.draw_networkx_nodes(
                graph_vis, pos, nodelist=self.marker_keys, node_color="r", node_size=100
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
                self.save_path,
                "visibility_graph-{0}-{1}.png".format(
                    len(self.visibility_graph), len(self.marker_keys)
                ),
            )
            plt.savefig(save_name)
            plt.clf()
