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
from observable import Observable


class MarkerLocation(model.storage.StorageItem):
    version = 1

    def __init__(self, frame_index_range, detections):
        self.frame_index_range = frame_index_range
        self.detections = detections

    @property
    def calculate_complete(self):
        return bool(self.detections)

    @staticmethod
    def from_tuple(tuple_):
        return MarkerLocation(*tuple_)

    @property
    def as_tuple(self):
        return self.frame_index_range, self.detections

    def __setitem__(self, frame_index, marker_location):
        self.detections[frame_index] = marker_location

    def __getitem__(self, frame_index):
        try:
            return self.detections[frame_index]["marker_detection"]
        except KeyError:
            return {}


class MarkerLocationStorage(model.storage.SingleFileStorage, Observable):
    def __init__(self, rec_dir, plugin, get_recording_index_range):
        super().__init__(rec_dir, plugin)
        self._get_recording_index_range = get_recording_index_range

        self._marker_locations = None
        self._load_from_disk()
        if not self._marker_locations:
            self._add_empty_marker_locations()

    def _add_empty_marker_locations(self):
        self.add(self.create_empty_marker_locations())

    def create_empty_marker_locations(self):
        return MarkerLocation(
            frame_index_range=self._get_recording_index_range(), detections={}
        )

    def add(self, marker_locations):
        self._marker_locations = marker_locations

    @property
    def _storage_file_name(self):
        return "marker_locations.msgpack"

    @property
    def item(self):
        return self._marker_locations

    @property
    def items(self):
        return [self._marker_locations] if self._marker_locations else []

    @property
    def _item_class(self):
        return MarkerLocation
