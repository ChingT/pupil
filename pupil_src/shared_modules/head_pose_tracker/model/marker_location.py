"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from head_pose_tracker import model


class MarkerLocation(model.storage.StorageItem):
    version = 1

    def __init__(self, marker_detection, frame_index, timestamp):
        self.marker_detection = marker_detection
        self.frame_index = frame_index
        self.timestamp = timestamp

    @staticmethod
    def from_tuple(tuple_):
        return MarkerLocation(*tuple_)

    @property
    def as_tuple(self):
        return self.marker_detection, self.frame_index, self.timestamp
