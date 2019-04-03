"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os

from head_pose_tracker import model
from observable import Observable


class MarkerLocations(model.StorageItem):
    version = 1

    def __init__(self, frame_index_range, result):
        self.frame_index_range = tuple(frame_index_range)
        self.result = result

    @staticmethod
    def from_tuple(tuple_):
        return MarkerLocations(*tuple_)

    @property
    def as_tuple(self):
        return self.frame_index_range, self.result

    @property
    def calculated(self):
        return bool(self.result)


class MarkerLocationStorage(model.Storage, Observable):
    def __init__(self, rec_dir, plugin, get_recording_index_range):
        super().__init__(rec_dir, plugin, get_recording_index_range)

    def _create_default_item(self):
        return MarkerLocations(
            frame_index_range=self._get_recording_index_range(), result={}
        )

    @property
    def _item_class(self):
        return MarkerLocations

    @property
    def _storage_folder_path(self):
        return os.path.join(self._rec_dir, "offline_data")

    @property
    def _storage_file_name(self):
        return "marker_locations.msgpack"
