import logging
import os

import numpy as np

import file_methods
from marker_tracker_3d import utils

logger = logging.getLogger(__name__)


class ModelOptimizationStorage:
    def __init__(self, save_path):
        self._model_save_path = os.path.join(save_path, "marker_tracker_3d_model")

        self._set_to_default_values()

        self._load_marker_tracker_3d_model_from_file()

    def _set_to_default_values(self):
        self.adding_marker_detections = True
        self.current_frame_id = 0

        self.frame_ids = []
        self.marker_ids = []

        self.all_novel_markers = []

        # camera_extrinsics_opt_dict: {frame id: optimized camera extrinsics (which is
        # composed of Rodrigues rotation vector and translation vector, which brings
        # points from the world coordinate system to the camera coordinate system)}
        self.camera_extrinsics_opt_dict = {}

        # marker_extrinsics_opt_dict: {marker id: optimized marker extrinsics}
        self.marker_extrinsics_opt_dict = {}

        # marker_points_3d_opt: {marker id: 3d points of 4 vertices of the marker
        # in the world coordinate system}.
        # it is updated according to marker_extrinsics_opt_dict by the function
        # extrinsics_to_marker_points_3d
        self.marker_points_3d_opt = {}

    def reset(self):
        self._set_to_default_values()

    def _load_marker_tracker_3d_model_from_file(self):
        marker_tracker_3d_model = file_methods.Persistent_Dict(self._model_save_path)

        self.marker_extrinsics_opt_dict = {
            marker_id: np.array(extrinsics)
            for marker_id, extrinsics in marker_tracker_3d_model.get(
                "marker_extrinsics_opt_dict", {}
            ).items()
        }
        self.marker_points_3d_opt = {
            marker_id: utils.extrinsics_to_marker_points_3d(extrinsics)[0]
            for marker_id, extrinsics in self.marker_extrinsics_opt_dict.items()
        }

        origin_marker_id = utils.find_origin_marker_id(self.marker_extrinsics_opt_dict)
        if origin_marker_id is not None:
            self.marker_ids = [origin_marker_id]

            logger.info(
                "marker tracker 3d model with {0} markers has been loaded from "
                "{1}".format(
                    len(self.marker_extrinsics_opt_dict), self._model_save_path
                )
            )

    def export_marker_tracker_3d_model(self):
        marker_tracker_3d_model = file_methods.Persistent_Dict(self._model_save_path)
        marker_tracker_3d_model["marker_extrinsics_opt_dict"] = {
            marker_id: extrinsics.tolist()
            for marker_id, extrinsics in self.marker_extrinsics_opt_dict.items()
        }
        marker_tracker_3d_model.save()

        logger.info(
            "marker tracker 3d model with {0} markers has been exported to {1}".format(
                len(self.marker_extrinsics_opt_dict), self._model_save_path
            )
        )
