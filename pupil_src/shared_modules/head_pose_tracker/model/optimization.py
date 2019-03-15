"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import numpy as np

from head_pose_tracker import model, worker


# this plugin does not care about the content of the result, it just receives it from
# the optimization routine and handles it to the camera localizer
class Optimization(model.storage.StorageItem):
    version = 1

    def __init__(
        self,
        unique_id,
        name,
        recording_uuid,
        frame_index_range,
        origin_marker_id=None,
        status="Not calculated yet",
        result=None,
    ):
        self.unique_id = unique_id
        self.name = name
        self.recording_uuid = recording_uuid
        self.frame_index_range = frame_index_range
        self.origin_marker_id = origin_marker_id
        self.status = status

        if result is not None:
            self.result = {
                marker_id: np.array(extrinsics)
                for marker_id, extrinsics in result.items()
            }
            self.result_vis = {
                marker_id: worker.utils.convert_marker_extrinsics_to_points_3d(
                    extrinsics
                )
                for marker_id, extrinsics in result.items()
            }
        else:
            self.result = None
            self.result_vis = {}

        self.centroid = np.zeros((3,), dtype=np.float32)
        self.calculate_centroid()

        self.optimize_camera_intrinsics = False

    def calculate_centroid(self):
        try:
            self.centroid = np.mean(self.result_vis.values(), axis=(0, 1))
        except IndexError:
            self.centroid = np.zeros((3,), dtype=np.float32)

    @staticmethod
    def from_tuple(tuple_):
        return Optimization(*tuple_)

    @property
    def as_tuple(self):
        if self.result:
            result = {key: value.tolist() for key, value in self.result.items()}
        else:
            result = {}
        return (
            self.unique_id,
            self.name,
            self.recording_uuid,
            self.frame_index_range,
            self.origin_marker_id,
            self.status,
            result,
        )
