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

import networkx as nx
import numpy as np

from head_pose_tracker import worker

KeyMarker = collections.namedtuple(
    "KeyMarker", ["frame_id", "marker_id", "verts", "bin"]
)


logger = logging.getLogger(__name__)


class ModelStorage:
    def __init__(self, predetermined_origin_marker_id=None):
        self.visibility_graph = nx.MultiGraph()

        # {frame id: optimized camera extrinsics (which is composed of Rodrigues
        # rotation vector and translation vector, which brings points from the world
        # coordinate system to the camera coordinate system)}
        self.frame_id_to_extrinsics_opt = {}

        # {marker id: optimized marker extrinsics}
        self.marker_id_to_extrinsics_opt = {}
        self.marker_id_to_points_3d_opt = {}

        self.origin_marker_id = predetermined_origin_marker_id

        self.all_key_markers = []

        n_bins_x = 2
        n_bins_y = 2
        self._bins_x = np.linspace(0, 1, n_bins_x + 1)[1:-1]
        self._bins_y = np.linspace(0, 1, n_bins_y + 1)[1:-1]

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
        else:
            self.marker_id_to_extrinsics_opt = {}
            self.marker_id_to_points_3d_opt = {}

    def save_key_markers(self, marker_id_to_detections, current_frame_id):
        key_markers = [
            KeyMarker(
                current_frame_id, marker_id, detection["verts"], self.get_bin(detection)
            )
            for marker_id, detection in marker_id_to_detections.items()
        ]
        self.all_key_markers += key_markers

        marker_ids = [marker.marker_id for marker in key_markers]
        key_edges = [
            (marker_id1, marker_id2, current_frame_id)
            for marker_id1, marker_id2 in list(it.combinations(marker_ids, 2))
        ]

        self.visibility_graph.add_edges_from(key_edges)

    def get_bin(self, detection):
        centroid = detection["centroid"]
        bin_x = int(np.digitize(centroid[0], self._bins_x))
        bin_y = int(np.digitize(centroid[1], self._bins_y))
        return bin_x, bin_y
