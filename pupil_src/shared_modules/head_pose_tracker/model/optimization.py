"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from collections import namedtuple

import numpy as np

from head_pose_tracker import model

# this plugin does not care about the content of the result, it just receives it from
# the optimization routine and handles it to the camera localizer
OptimizationResult = namedtuple(
    "OptimizationResult", ["mapping_plugin_name", "mapper_args"]
)


class Optimization(model.storage.StorageItem):
    version = 1

    def __init__(
        self,
        unique_id,
        name,
        recording_uuid,
        frame_index_range,
        status="Not calculated yet",
        result=None,
    ):
        self.unique_id = unique_id
        self.name = name
        self.recording_uuid = recording_uuid
        self.frame_index_range = frame_index_range
        self.status = status

        if result:
            self.result = {key: np.array(value) for key, value in result.items()}
        else:
            self.result = None

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
            self.status,
            result,
        )
