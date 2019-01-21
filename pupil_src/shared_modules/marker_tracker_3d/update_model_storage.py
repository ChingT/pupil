import logging

from marker_tracker_3d import utils
from observable import Observable

logger = logging.getLogger(__name__)


class UpdateModelStorage(Observable):
    def __init__(self, model_storage):
        self._model_storage = model_storage

    def run(self, model_opt_result):
        """ process the results of optimization; update frame_id_to_extrinsics_opt,
        marker_id_to_extrinsics_opt and marker_id_to_points_3d_opt """

        if model_opt_result:
            self._update_extrinsics_opt(
                model_opt_result.frame_id_to_extrinsics,
                model_opt_result.marker_id_to_extrinsics,
            )
            self._discard_failed_frames(model_opt_result.frame_ids_failed)
        self.on_update_model_storage_done()

    def _update_extrinsics_opt(self, frame_id_to_extrinsics, marker_id_to_extrinsics):
        self._model_storage.frame_id_to_extrinsics_opt = frame_id_to_extrinsics

        for marker_id, extrinsics in marker_id_to_extrinsics.items():
            self._model_storage.marker_id_to_extrinsics_opt[marker_id] = extrinsics
            self._model_storage.marker_id_to_points_3d_opt[
                marker_id
            ] = utils.convert_marker_extrinsics_to_points_3d(extrinsics)

        logger.debug(
            "{} markers have been registered and updated".format(
                len(self._model_storage.marker_id_to_extrinsics_opt)
            )
        )

    def _discard_failed_frames(self, frame_ids_failed):
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

        self._model_storage.all_novel_markers = [
            marker
            for marker in self._model_storage.all_novel_markers
            if marker.frame_id not in frame_ids_failed
        ]

        logger.debug("discard_failed_frames {0}".format(frame_ids_failed))

    def on_update_model_storage_done(self):
        pass
