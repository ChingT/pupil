"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

from head_pose_tracker import model
from observable import Observable

logger = logging.getLogger(__name__)


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


class MarkerLocationStorage(model.SingleFileStorage, Observable):
    def __init__(self, rec_dir, plugin):
        super().__init__(rec_dir, plugin)
        self._marker_locations = {}
        self._load_from_disk()

    def add(self, marker_location):
        self._marker_locations[marker_location.frame_index] = marker_location

    def get_or_none(self, frame_index):
        return self._marker_locations.get(frame_index, None)

    @property
    def _storage_file_name(self):
        return "marker_locations.msgpack"

    @property
    def _item_class(self):
        return MarkerLocation

    @property
    def items(self):
        return self._marker_locations.values()
