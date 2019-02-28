import collections
import itertools as it
import logging
import os

import numpy as np

import file_methods
from marker_tracker_3d import worker

logger = logging.getLogger(__name__)
KeyMarker = collections.namedtuple(
    "KeyMarker", ["frame_id", "marker_id", "verts", "bin"]
)


class ControllerStorage:
    def __init__(self, save_path):
        self._all_camera_poses_path = os.path.join(save_path, "all_camera_poses")

        self._n_bins_x = 4
        self._n_bins_y = 2
        self._bins_x = np.linspace(0, 1, self._n_bins_x + 1)[1:-1]
        self._bins_y = np.linspace(0, 1, self._n_bins_y + 1)[1:-1]

        self._set_to_default_values()

    def _set_to_default_values(self):
        self._not_localized_count = 0

        # for export
        self.all_camera_poses = {}
        # for drawing in 2d window
        self.marker_id_to_detections = {}

        # for drawing in 3d window
        self.all_camera_traces = []
        self.camera_pose_matrix = None
        self._camera_extrinsics = None
        self.camera_extrinsics = None

        self.all_key_markers = []
        self.key_edges_queue = []
        self.key_markers_queue = []
        self.key_markers_bins = {
            (x, y): [] for x in range(self._n_bins_x) for y in range(self._n_bins_y)
        }

    def reset(self):
        self._set_to_default_values()

    def save_observation(
        self, marker_id_to_detections, camera_extrinsics, current_frame_id
    ):
        self.marker_id_to_detections = marker_id_to_detections
        self.camera_extrinsics = camera_extrinsics

        self._save_camera_pose(camera_extrinsics, current_frame_id)

    @property
    def marker_id_to_detections(self):
        return self._marker_id_to_detections

    @marker_id_to_detections.setter
    def marker_id_to_detections(self, _marker_id_to_detections):
        for marker_id in _marker_id_to_detections.keys():
            centroid = _marker_id_to_detections[marker_id]["centroid"]
            bin_x = int(np.digitize(centroid[0], self._bins_x))
            bin_y = int(np.digitize(centroid[1], self._bins_y))
            _marker_id_to_detections[marker_id]["bin"] = (bin_x, bin_y)

        self._marker_id_to_detections = _marker_id_to_detections

    def save_key_markers(self, marker_id_to_detections, current_frame_id):
        key_markers = [
            KeyMarker(current_frame_id, marker_id, detection["verts"], detection["bin"])
            for marker_id, detection in marker_id_to_detections.items()
        ]
        self.key_markers_queue += key_markers

        marker_ids = [marker.marker_id for marker in key_markers]
        key_edges = [
            (marker_id1, marker_id2, current_frame_id)
            for marker_id1, marker_id2 in list(it.combinations(marker_ids, 2))
        ]
        self.key_edges_queue += key_edges

        for marker_id, detection in marker_id_to_detections.items():
            self.key_markers_bins[detection["bin"]].append(
                (current_frame_id, marker_id)
            )

    @property
    def camera_extrinsics(self):
        return self._camera_extrinsics

    @camera_extrinsics.setter
    def camera_extrinsics(self, _camera_extrinsics):
        if _camera_extrinsics is not None:
            self._camera_extrinsics = _camera_extrinsics

            camera_poses = worker.utils.get_camera_pose(_camera_extrinsics)
            self.camera_pose_matrix = worker.utils.convert_extrinsic_to_matrix(
                camera_poses
            )
            self.all_camera_traces.append(camera_poses[3:6])
            self.all_camera_poses[self.current_frame_id] = camera_poses

            self.frame_id_to_extrinsics_all[self.current_frame_id] = _camera_extrinsics
            self._not_localized_count = 0
        else:
            # Do not set camera_extrinsics to None to ensure
            # a decent initial guess for the next solvePnP call;
            # if there are multiple frames which could not be localized,
            # then set camera_extrinsics to None
            if self._not_localized_count >= 3:
                self._camera_extrinsics = None
                self.camera_pose_matrix = None

            self._not_localized_count += 1
            self.all_camera_traces.append(np.full((3,), np.nan))
