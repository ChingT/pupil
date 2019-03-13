"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import player_methods as pm
from head_pose_tracker import model


class CameraLocalizer(model.storage.StorageItem):
    version = 1

    def __init__(
        self,
        unique_id,
        name,
        localization_index_range,
        status="Not calculated yet",
        pose=[],
        pose_ts=[],
        pose_bisector=pm.Bisector(),
    ):
        self.unique_id = unique_id
        self.name = name
        self.localization_index_range = tuple(localization_index_range)
        self.status = status
        self.pose = pose
        self.pose_ts = pose_ts
        self.pose_bisector = pose_bisector

    @property
    def calculate_complete(self):
        # we cannot just use `self.pose and self.pose_ts` because this ands the arrays
        return len(self.pose) > 0 and len(self.pose_ts) > 0

    @staticmethod
    def from_tuple(tuple_):
        return CameraLocalizer(*tuple_)

    @property
    def as_tuple(self):
        return (self.unique_id, self.name, self.localization_index_range, self.status)
