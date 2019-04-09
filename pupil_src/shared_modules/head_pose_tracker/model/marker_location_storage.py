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

import file_methods as fm
import player_methods as pm
from head_pose_tracker import model
from observable import Observable


class MarkerLocations(model.StorageItem):
    version = 1

    def __init__(self, frame_index_range, calculated_frame_indices):
        self.frame_index_range = tuple(frame_index_range)
        self.calculated_frame_indices = calculated_frame_indices

        self.markers_bisector = pm.Mutable_Bisector()

    @staticmethod
    def from_tuple(tuple_):
        return MarkerLocations(*tuple_)

    @property
    def as_tuple(self):
        return self.frame_index_range, self.calculated_frame_indices

    @property
    def calculated(self):
        return bool(self.markers_bisector)


class MarkerLocationStorage(model.Storage, Observable):
    def __init__(self, rec_dir, plugin, get_recording_index_range):
        super().__init__(rec_dir, plugin, get_recording_index_range)

    def _create_default_item(self):
        return MarkerLocations(
            frame_index_range=self._get_recording_index_range(),
            calculated_frame_indices=[],
        )

    def save_to_disk(self):
        # this will save everything except markers and markers_ts
        super().save_to_disk()

        self._save_markers_and_ts_to_disk()

    def _save_markers_and_ts_to_disk(self):
        directory = self._storage_folder_path
        file_name = self._marker_locations_file_name
        with fm.PLData_Writer(directory, file_name) as writer:
            for marker_ts, marker in zip(
                self.item.markers_bisector.timestamps, self.item.markers_bisector.data
            ):
                writer.append_serialized(
                    marker_ts, topic="marker", datum_serialized=marker.serialized
                )

    def load_from_disk(self, file_path):
        # this will load everything except pose and pose_ts
        super().load_from_disk(file_path)

        if self.item:
            self._load_markers_and_ts_to_disk()

    def _load_markers_and_ts_to_disk(self):
        directory = self._storage_folder_path
        file_name = self._marker_locations_file_name
        pldata = fm.load_pldata_file(directory, file_name)
        self.item.markers_bisector = pm.Mutable_Bisector(pldata.data, pldata.timestamps)

    @property
    def _item_class(self):
        return MarkerLocations

    @property
    def _storage_folder_path(self):
        return os.path.join(self._rec_dir, "offline_data")

    @property
    def _storage_file_name(self):
        return "marker_locations.msgpack"

    @property
    def _marker_locations_file_name(self):
        return "marker_locations"
