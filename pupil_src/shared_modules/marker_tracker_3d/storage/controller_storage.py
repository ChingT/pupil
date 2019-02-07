import logging
import os

import numpy as np

from marker_tracker_3d import worker
from observable import Observable

logger = logging.getLogger(__name__)


class ControllerStorage(Observable):
    def __init__(self, min_marker_perimeter, save_path):
        self.min_marker_perimeter = min_marker_perimeter  # adjustable in UI

        self.save_path = save_path

        self._set_to_default_values()

    def _set_to_default_values(self):
        # for drawing in 2d window
        self.marker_id_to_detections = {}

        # for drawing in 3d window
        self.all_camera_traces = []
        self.camera_pose_matrix = None
        self._camera_extrinsics = None
        self.camera_extrinsics = None

    def reset(self):
        self._set_to_default_values()

    def save_observation(self, marker_id_to_detections, camera_extrinsics):
        self.marker_id_to_detections = marker_id_to_detections
        self.camera_extrinsics = camera_extrinsics

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
    def camera_extrinsics(self):
        return self._camera_extrinsics

    @camera_extrinsics.setter
    def camera_extrinsics(self, camera_extrinsics_new):
        if camera_extrinsics_new is not None:
            self._camera_extrinsics = camera_extrinsics_new
            self.camera_pose_matrix = worker.utils.get_camera_pose_matrix(
                camera_extrinsics_new
            )
            self.all_camera_traces.append(
                worker.utils.get_camera_trace(self.camera_pose_matrix)
            )
        else:
            # Do not set camera_extrinsics to None to ensure
            # a decent initial guess for the next solvePnP call
            self.camera_pose_matrix = None
            self.all_camera_traces.append(np.full((3,), np.nan))

    def get_init_dict(self):
        d = {"min_marker_perimeter": self.min_marker_perimeter}
        return d
