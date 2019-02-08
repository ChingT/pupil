import logging

from marker_tracker_3d import worker
from observable import Observable

logger = logging.getLogger(__name__)


class UpdateModelStorage(Observable):
    def __init__(self, model_storage, camera_intrinsics):
        self._model_storage = model_storage
        self._camera_intrinsics = camera_intrinsics

    def run(self, model_opt_result):
        """ process the results of optimization; update frame_id_to_extrinsics_opt,
        marker_id_to_extrinsics_opt and marker_id_to_points_3d_opt """

        if not model_opt_result:
            return

        self._update_extrinsics_opt(
            model_opt_result.frame_id_to_extrinsics,
            model_opt_result.marker_id_to_extrinsics,
        )
        self._discard_failed_novel_markers(
            model_opt_result.frame_ids_failed, model_opt_result.marker_ids_failed
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

        logger.debug(
            "{} markers have been registered and updated".format(
                len(self._model_storage.marker_id_to_extrinsics_opt)
            )
        )

    def _discard_failed_novel_markers(self, frame_ids_failed, marker_ids_failed):
        if not frame_ids_failed or not marker_ids_failed:
            return

        redundant_edges = [
            (node, neighbor, frame_id)
            for node, neighbor, frame_id in self._model_storage.visibility_graph.edges(
                keys=True
            )
            if frame_id in frame_ids_failed
            and (node in marker_ids_failed or neighbor in marker_ids_failed)
        ]
        self._model_storage.visibility_graph.remove_edges_from(redundant_edges)

        self._model_storage.all_novel_markers = [
            marker
            for marker in self._model_storage.all_novel_markers
            if not (
                marker.frame_id in frame_ids_failed
                and marker.marker_id in marker_ids_failed
            )
        ]

        logger.debug("discard_failed_frames {0}".format(frame_ids_failed))

    # TODO: debug only; to be removed
    def run_init(self, model_init_result):
        if model_init_result:
            self._update_extrinsics_init(model_init_result.marker_id_to_extrinsics)

    # TODO: debug only; to be removed
    def _update_extrinsics_init(self, marker_id_to_extrinsics_init):
        self._model_storage.marker_id_to_points_3d_init = {}
        for marker_id, extrinsics in marker_id_to_extrinsics_init.items():
            self._model_storage.marker_id_to_points_3d_init[
                marker_id
            ] = worker.utils.convert_marker_extrinsics_to_points_3d(extrinsics)
