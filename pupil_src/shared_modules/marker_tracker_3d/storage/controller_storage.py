import logging
import os

import numpy as np

import file_methods
from marker_tracker_3d import worker
from observable import Observable

logger = logging.getLogger(__name__)


class ControllerStorage(Observable):
    def __init__(self, min_marker_perimeter, save_path):
        self.min_marker_perimeter = min_marker_perimeter  # adjustable in UI

        self._all_camera_poses_path = os.path.join(save_path, "all_camera_poses")

        self._set_to_default_values()

    def _set_to_default_values(self):
        self.current_frame_id = 0

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

    def export_all_camera_poses(self):
        all_camera_poses_object = {
            frame_id: camera_poses.tolist()
            for frame_id, camera_poses in self.all_camera_poses.items()
        }
        file_methods.save_object(all_camera_poses_object, self._all_camera_poses_path)

        logger.info(
            "camera trace from {0} frames has been exported to {1}".format(
                len(all_camera_poses_object), self._all_camera_poses_path
            )
        )

    @property
    def camera_extrinsics(self):
        return self._camera_extrinsics

    @camera_extrinsics.setter
    def camera_extrinsics(self, _camera_extrinsics):
        if _camera_extrinsics is not None:
            self._camera_extrinsics = _camera_extrinsics

            camera_poses = worker.utils.get_camera_pose(_camera_extrinsics)
            self.camera_pose_matrix = worker.utils.get_extrinsic_matrix(camera_poses)
            self.all_camera_traces.append(camera_poses[3:6])
            self.all_camera_poses[self.current_frame_id] = camera_poses

            self.frame_id_to_extrinsics_all[self.current_frame_id] = _camera_extrinsics
        else:
            # Do not set camera_extrinsics to None to ensure
            # a decent initial guess for the next solvePnP call
            self.camera_pose_matrix = None
            self.all_camera_traces.append(np.full((3,), np.nan))

    def get_init_dict(self):
        d = {"min_marker_perimeter": self.min_marker_perimeter}
        return d
