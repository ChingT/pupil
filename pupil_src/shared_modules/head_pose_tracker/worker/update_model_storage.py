"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from head_pose_tracker import worker


class UpdateModelStorage:
    def __init__(self, model_storage):
        self._model_storage = model_storage

    def run(self, model_opt_result):
        """ process the results of markers_3d_model; update frame_id_to_extrinsics_opt,
        marker_id_to_extrinsics_opt and marker_id_to_points_3d_opt """

        if not model_opt_result:
            return

        self._update_extrinsics_opt(
            model_opt_result.frame_id_to_extrinsics,
            model_opt_result.marker_id_to_extrinsics,
        )

        self._discard_failed_key_markers(model_opt_result.frame_ids_failed)

    def _update_extrinsics_opt(self, frame_id_to_extrinsics, marker_id_to_extrinsics):
        self._model_storage.frame_id_to_extrinsics_opt.update(frame_id_to_extrinsics)

        for marker_id, extrinsics in marker_id_to_extrinsics.items():
            self._model_storage.marker_id_to_extrinsics_opt[marker_id] = extrinsics
            self._model_storage.marker_id_to_points_3d_opt[
                marker_id
            ] = worker.utils.convert_marker_extrinsics_to_points_3d(extrinsics)

    def _discard_failed_key_markers(self, frame_ids_failed):
        if not frame_ids_failed:
            return

        redundant_edges = [
            (node, neighbor, frame_id)
            for node, neighbor, frame_id in self._model_storage.visibility_graph.edges(
                keys=True
            )
            if frame_id in frame_ids_failed
        ]
        self._model_storage.visibility_graph.remove_edges_from(redundant_edges)

        self._model_storage.all_key_markers = [
            marker
            for marker in self._model_storage.all_key_markers
            if marker.frame_id not in frame_ids_failed
        ]
