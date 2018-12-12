import logging

import numpy as np

from marker_tracker_3d import utils

logger = logging.getLogger(__name__)


class ControllerStorage:
    def __init__(self, save_path):
        self.save_path = save_path

        # For drawing in UI window
        self.marker_detections = {}
        self.camera_pose_matrix = None
        self.all_camera_traces = []
        self.camera_extrinsics = None

    def reset(self):
        self.marker_detections = {}
        self.camera_pose_matrix = None
        self.all_camera_traces = []
        self.camera_extrinsics = None

    def export_data(self):
        utils.save_array(self.save_path, "all_camera_traces", self.all_camera_traces)
        logger.info("camera trace has been exported to {}".format(self.save_path))

    @property
    def camera_extrinsics(self):
        return self.__camera_extrinsics

    @camera_extrinsics.setter
    def camera_extrinsics(self, camera_extrinsics_new):
        self.__camera_extrinsics = camera_extrinsics_new
        if camera_extrinsics_new is not None:
            self.camera_pose_matrix = utils.get_camera_pose_matrix(
                camera_extrinsics_new
            )
            self.all_camera_traces.append(
                utils.get_camera_trace(self.camera_pose_matrix)
            )
        else:
            self.camera_pose_matrix = None
            self.all_camera_traces.append(np.full((3,), np.nan))
