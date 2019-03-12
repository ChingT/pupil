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
# the optimization routine and handles it to the gaze mapper
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
        mapping_method,
        frame_index_range,
        minimum_confidence,
        status="Not calculated yet",
        is_offline_optimization=True,
        result=None,
    ):
        self.unique_id = unique_id
        self.name = name
        self.recording_uuid = recording_uuid
        self.mapping_method = mapping_method
        self.frame_index_range = frame_index_range
        self.minimum_confidence = minimum_confidence
        self.status = status
        self.is_offline_optimization = is_offline_optimization

        if result:
            self.result = {key: np.array(value) for key, value in result.items()}
        else:
            self.result = None

    @staticmethod
    def from_tuple(tuple_):
        return Optimization(*tuple_)

    @property
    def as_tuple(self):
        result = {key: value.tolist() for key, value in self.result.items()}
        return (
            self.unique_id,
            self.name,
            self.recording_uuid,
            self.mapping_method,
            self.frame_index_range,
            self.minimum_confidence,
            self.status,
            self.is_offline_optimization,
            result,
        )
