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

import networkx as nx

logger = logging.getLogger(__name__)


DataForModelInit = collections.namedtuple(
    "DataForModelInit",
    [
        "key_markers",
        "frame_id_to_extrinsics_prv",
        "marker_id_to_extrinsics_prv",
        "frame_ids_to_be_optimized",
        "marker_ids_to_be_optimized",
    ],
)


class PrepareForModelUpdate:
    def __init__(self, controller_storage, model_storage):
        self._controller_storage = controller_storage
        self._model_storage = model_storage

    def run(self):
        marker_ids_to_be_optimized = self._get_marker_ids_to_be_optimized()
        frame_ids_to_be_optimized = self._get_frame_ids_to_be_optimized(
            marker_ids_to_be_optimized
        )

        if not frame_ids_to_be_optimized:
            return None

        key_markers = [
            marker
            for marker in self._controller_storage.all_key_markers
            if (
                marker.frame_id in frame_ids_to_be_optimized
                and marker.marker_id in marker_ids_to_be_optimized
            )
        ]
        frame_id_to_extrinsics_prv = self._get_frame_id_to_extrinsics_prv(
            frame_ids_to_be_optimized
        )
        marker_id_to_extrinsics_prv = self._get_marker_id_to_extrinsics_prv()

        data_for_model_init = DataForModelInit(
            key_markers,
            frame_id_to_extrinsics_prv,
            marker_id_to_extrinsics_prv,
            frame_ids_to_be_optimized,
            marker_ids_to_be_optimized,
        )
        return data_for_model_init

    def _get_marker_ids_to_be_optimized(self):
        try:
            connected_component = nx.node_connected_component(
                self._controller_storage.visibility_graph,
                self._model_storage.origin_marker_id,
            )
        except KeyError:
            self._set_coordinate_system(self._controller_storage.visibility_graph.nodes)
            return []

        marker_ids_to_be_optimized = [self._model_storage.origin_marker_id] + list(
            connected_component - {self._model_storage.origin_marker_id}
        )
        return marker_ids_to_be_optimized

    def _get_frame_ids_to_be_optimized(self, marker_ids_to_be_optimized):
        frame_ids_to_be_optimized = set(
            marker.frame_id
            for marker in self._controller_storage.all_key_markers
            if marker.marker_id in marker_ids_to_be_optimized
        )
        return list(frame_ids_to_be_optimized)

    def _set_coordinate_system(self, all_seen_markers_id):
        try:
            origin_marker_id = list(all_seen_markers_id)[0]
        except IndexError:
            pass
        else:
            self._model_storage.origin_marker_id = origin_marker_id

    def _get_frame_id_to_extrinsics_prv(self, frame_ids_to_be_optimized):
        all_frame_ids = (
            set(frame_ids_to_be_optimized)
            & self._model_storage.frame_id_to_extrinsics_opt.keys()
        )
        frame_id_to_extrinsics_prv = {
            frame_id: self._model_storage.frame_id_to_extrinsics_opt[frame_id]
            for frame_id in all_frame_ids
        }
        return frame_id_to_extrinsics_prv

    def _get_marker_id_to_extrinsics_prv(self):
        # Do not need to use .copy(), since it will be copied to bg_task
        return self._model_storage.marker_id_to_extrinsics_opt
