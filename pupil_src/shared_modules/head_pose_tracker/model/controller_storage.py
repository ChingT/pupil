"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import logging

import networkx as nx

from head_pose_tracker import worker

logger = logging.getLogger(__name__)


KeyMarker = collections.namedtuple(
    "KeyMarker", ["frame_id", "marker_id", "verts", "bin"]
)


class ControllerStorage:
    def __init__(self):
        self._set_to_default_values()

    def _set_to_default_values(self):
        self._not_localized_count = 0

        self.visibility_graph = nx.MultiGraph()

        # for drawing in 2d window
        self.marker_id_to_detections = {}

        # for drawing in 3d window
        # self.recent_camera_traces = collections.deque(maxlen=300)
        self.camera_pose_matrix = None
        self._camera_extrinsics = None
        self.camera_extrinsics = None

        self.all_key_markers = []
        self.all_key_edges = []
        self.n_key_markers_processed = 0

    def reset(self):
        self._set_to_default_values()

    def update_current_marker_id_to_detections(self, marker_id_to_detections):
        if marker_id_to_detections is None:
            self.marker_id_to_detections = {}
        else:
            self.marker_id_to_detections = marker_id_to_detections

    @property
    def marker_id_to_detections(self):
        return self._marker_id_to_detections

    @marker_id_to_detections.setter
    def marker_id_to_detections(self, _marker_id_to_detections):
        self._marker_id_to_detections = _marker_id_to_detections

    def update_current_camera_extrinsics(self, camera_extrinsics):
        self.camera_extrinsics = camera_extrinsics

    def update_current_camera_pose(self, camera_extrinsics):
        camera_poses = worker.utils.get_camera_pose(camera_extrinsics)
        self.recent_camera_traces.append(camera_poses[3:6])
        self.camera_pose_matrix = worker.utils.convert_extrinsic_to_matrix(camera_poses)

    @property
    def camera_extrinsics(self):
        return self._camera_extrinsics

    @camera_extrinsics.setter
    def camera_extrinsics(self, _camera_extrinsics):
        if _camera_extrinsics is not None:
            self._camera_extrinsics = _camera_extrinsics
            self._not_localized_count = 0
        else:
            # Do not set camera_extrinsics to None to ensure
            # a decent initial guess for the next solvePnP call;
            # except when there are multiple frames which could not be localized,
            # then set camera_extrinsics to None
            if self._not_localized_count >= 3:
                self._camera_extrinsics = None
            self._not_localized_count += 1
