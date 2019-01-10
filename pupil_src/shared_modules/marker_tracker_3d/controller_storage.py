import logging
import os

import numpy as np

from marker_tracker_3d import utils

logger = logging.getLogger(__name__)


class ControllerStorage:
    def __init__(self, save_path):
        self.save_path = save_path

        # For drawing in UI window
        self.current_marker_detections = {}
        self.current_camera_pose_matrix = None
        self.current_camera_extrinsics = None
        self.all_camera_traces = []

    def reset(self):
        self.current_marker_detections = {}
        self.current_camera_pose_matrix = None
        self.current_camera_extrinsics = None
        self.all_camera_traces = []

    def export_camera_traces(self):
        np.save(
            os.path.join(self.save_path, "all_camera_traces"), self.all_camera_traces
        )

        logger.info(
            "camera trace from {0} frames has been exported to {1}".format(
                len(self.all_camera_traces),
                os.path.join(self.save_path, "all_camera_traces"),
            )
        )

    @property
    def current_camera_extrinsics(self):
        return self.__camera_extrinsics

    @current_camera_extrinsics.setter
    def current_camera_extrinsics(self, camera_extrinsics_new):
        self.__camera_extrinsics = camera_extrinsics_new
        if camera_extrinsics_new is not None:
            self.current_camera_pose_matrix = utils.get_camera_pose_matrix(
                camera_extrinsics_new
            )
            self.all_camera_traces.append(
                utils.get_camera_trace(self.current_camera_pose_matrix)
            )
        else:
            self.current_camera_pose_matrix = None
            try:
                self.all_camera_traces.append(np.full((3,), np.nan))
            except AttributeError:
                return
