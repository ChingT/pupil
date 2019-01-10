import logging
import os

import numpy as np

import file_methods
from marker_tracker_3d import utils

logger = logging.getLogger(__name__)


class ModelState:
    def __init__(self, save_path):
        self.save_path = save_path

        self.current_frame_id = 0

        self.frames_id = []
        self.markers_id = []

        self.all_novel_markers = []

        self.camera_extrinsics_opt = {}  # {frame id: camera extrinsics(list)}
        self.marker_extrinsics_opt = {}  # {marker id: marker extrinsics(list)}
        self.marker_points_3d_opt = {}

        self.model = None
        self.load_marker_tracker_3d_model_from_file()

    def load_marker_tracker_3d_model_from_file(self):
        self.model = file_methods.Persistent_Dict(
            os.path.join(self.save_path, "marker_tracker_3d_model")
        )
        self.marker_extrinsics_opt = {
            k: np.array(v)
            for k, v in self.model.get("marker_extrinsics_opt", {}).items()
        }

        try:
            self.markers_id = [list(self.marker_extrinsics_opt.keys())[0]]
        except IndexError:
            self.markers_id = []
        else:
            self.marker_points_3d_opt = {
                k: utils.params_to_points_3d(v)[0]
                for k, v in self.marker_extrinsics_opt.items()
            }

            logger.info(
                "marker tracker 3d model with {0} markers has been loaded from {1}".format(
                    len(self.marker_extrinsics_opt), self.model.file_path
                )
            )

    def export_marker_tracker_3d_model(self):
        self.model["marker_extrinsics_opt"] = {
            k: v.tolist() for k, v in self.marker_extrinsics_opt.items()
        }
        self.model.save()

        logger.info(
            "marker tracker 3d model with {0} markers has been exported to {1}".format(
                len(self.marker_extrinsics_opt), self.model.file_path
            )
        )

    def reset(self):
        self.current_frame_id = 0

        self.frames_id = []
        self.markers_id = []

        self.all_novel_markers = []
        self.camera_extrinsics_opt = {}
        self.marker_extrinsics_opt = {}
        self.marker_points_3d_opt = {}
