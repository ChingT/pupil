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


class CameraLocalizer(model.StorageItem):
    version = 1

    def __init__(self, frame_index_range, status="Not calculated yet"):
        self.frame_index_range = tuple(frame_index_range)
        self.status = status

        # for visualization of pose data
        self.pose_bisector = pm.Bisector()

        self.show_camera_trace = True

    @staticmethod
    def from_tuple(tuple_):
        return CameraLocalizer(*tuple_)

    @property
    def as_tuple(self):
        return self.frame_index_range, self.status

    @property
    def calculated(self):
        return bool(self.pose_bisector)


class CameraLocalizerStorage(model.Storage, Observable):
    def __init__(self, rec_dir, plugin, get_recording_index_range):
        super().__init__(rec_dir, plugin, get_recording_index_range)

    def _create_default_item(self):
        return CameraLocalizer(frame_index_range=self._get_recording_index_range())

    def save_to_disk(self):
        # this will save everything except pose and pose_ts
        super().save_to_disk()

        self._save_pose_and_ts_to_disk()

    def _save_pose_and_ts_to_disk(self):
        directory = self._storage_folder_path
        file_name = self._camera_localization_file_name
        with fm.PLData_Writer(directory, file_name) as writer:
            for pose_ts, pose in zip(
                self.item.pose_bisector.timestamps, self.item.pose_bisector.data
            ):
                writer.append_serialized(
                    pose_ts, topic="pose", datum_serialized=pose.serialized
                )

    def load_from_disk(self, file_path):
        # this will load everything except pose and pose_ts
        super().load_from_disk(file_path)

        if self.item:
            self._load_pose_and_ts_from_disk()

    def _load_pose_and_ts_from_disk(self):
        directory = self._storage_folder_path
        file_name = self._camera_localization_file_name
        pldata = fm.load_pldata_file(directory, file_name)
        self.item.pose_bisector = pm.Bisector(pldata.data, pldata.timestamps)

    @property
    def _item_class(self):
        return CameraLocalizer

    @property
    def _storage_folder_path(self):
        return os.path.join(self._rec_dir, "offline_data")

    @property
    def _storage_file_name(self):
        return "camera_localizers.msgpack"

    @property
    def _camera_localization_file_name(self):
        return "camera_poses"
