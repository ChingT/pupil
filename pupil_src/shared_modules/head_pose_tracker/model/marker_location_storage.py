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


class MarkerLocationStorage(model.SingleFileStorage, Observable):
    def __init__(self, rec_dir, plugin):
        super().__init__(rec_dir, plugin)
        self._marker_locations = {}
        self._load_from_disk()

    def add(self, marker_location):
        self._marker_locations[marker_location.frame_index] = marker_location

    def get_or_none(self, frame_index):
        return self._marker_locations.get(frame_index, None)

    def delete(self, marker_location):
        del self._marker_locations[marker_location.frame_index]

    def delete_all(self):
        self._marker_locations.clear()

    @property
    def _storage_file_name(self):
        return "marker_locations.msgpack"

    @property
    def _item_class(self):
        return model.MarkerLocation

    @property
    def items(self):
        return self._marker_locations.values()
