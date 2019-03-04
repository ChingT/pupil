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
from observable import Observable


class UpdateModelStorage(Observable):
    def __init__(self, controller_storage, model_storage, camera_intrinsics):
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._camera_intrinsics = camera_intrinsics

    def run(self, model_opt_result):
        """ process the results of optimization; update frame_id_to_extrinsics_opt,
        marker_id_to_extrinsics_opt and marker_id_to_points_3d_opt """

        if not model_opt_result:
            return

        self._discard_failed_key_markers(model_opt_result.frame_ids_failed)

        self._update_extrinsics_opt(
            model_opt_result.frame_id_to_extrinsics,
            model_opt_result.marker_id_to_extrinsics,
        )

        if model_opt_result.camera_matrix is not None:
            self._camera_intrinsics.update_camera_matrix(model_opt_result.camera_matrix)
            self._camera_intrinsics.update_dist_coefs(model_opt_result.dist_coefs)

    def _update_extrinsics_opt(self, frame_id_to_extrinsics, marker_id_to_extrinsics):
        self._model_storage.frame_id_to_extrinsics_opt.update(frame_id_to_extrinsics)

        for marker_id, extrinsics in marker_id_to_extrinsics.items():
            self._model_storage.marker_id_to_extrinsics_opt[marker_id] = extrinsics
            self._model_storage.marker_id_to_points_3d_opt[
                marker_id
            ] = worker.utils.convert_marker_extrinsics_to_points_3d(extrinsics)

        self._model_storage.calculate_points_3d_centroid()

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

        self._controller_storage.all_key_markers = [
            marker
            for marker in self._controller_storage.all_key_markers
            if marker.frame_id not in frame_ids_failed
        ]

    # TODO: debug only; to be removed
    def run_init(self, model_init_result):
        if model_init_result:
            self._update_extrinsics_init(model_init_result.marker_id_to_extrinsics)

    # TODO: debug only; to be removed
    def _update_extrinsics_init(self, marker_id_to_extrinsics_init):
        for marker_id, extrinsics in marker_id_to_extrinsics_init.items():
            self._model_storage.marker_id_to_points_3d_init[
                marker_id
            ] = worker.utils.convert_marker_extrinsics_to_points_3d(extrinsics)
