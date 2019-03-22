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
import make_unique
import player_methods as pm
from head_pose_tracker import model
from observable import Observable


class CameraLocalizer(model.storage.StorageItem):
    version = 1

    def __init__(
        self, unique_id, name, localization_index_range, status="Not calculated yet"
    ):
        self.unique_id = unique_id
        self.name = name
        self.localization_index_range = tuple(localization_index_range)
        self.status = status

        # for visualization of pose data
        self.pose_bisector = pm.Bisector()

        self.show_camera_trace = True

    @property
    def calculate_complete(self):
        return bool(self.pose_bisector)

    @staticmethod
    def from_tuple(tuple_):
        return CameraLocalizer(*tuple_)

    @property
    def as_tuple(self):
        return self.unique_id, self.name, self.localization_index_range, self.status


class CameraLocalizerStorage(model.storage.SingleFileStorage, Observable):
    def __init__(self, rec_dir, plugin, get_recording_index_range):
        super().__init__(rec_dir, plugin)
        self._get_recording_index_range = get_recording_index_range

        self._camera_localizer = None
        self._load_from_disk()
        if not self._camera_localizer:
            self._add_default_camera_localizer()

    def _add_default_camera_localizer(self):
        self.add(self.create_default_camera_localizer())

    def create_default_camera_localizer(self):
        return CameraLocalizer(
            unique_id=CameraLocalizer.create_new_unique_id(),
            name=make_unique.by_number_at_end("Camera Localizer", self.item_names),
            localization_index_range=self._get_recording_index_range(),
        )

    def add(self, camera_localizer):
        self._camera_localizer = camera_localizer

    def save_to_disk(self):
        # this will save everything except pose and pose_ts
        super().save_to_disk()

        self._save_pose_and_ts_to_disk()

    def _save_pose_and_ts_to_disk(self):
        directory = self._camera_localizations_directory
        os.makedirs(directory, exist_ok=True)
        file_name = self._camera_localization_file_name(self._camera_localizer)
        with fm.PLData_Writer(directory, file_name) as writer:
            for pose_ts, pose in zip(
                self._camera_localizer.pose_bisector.timestamps,
                self._camera_localizer.pose_bisector.data,
            ):
                writer.append_serialized(
                    pose_ts, topic="pose", datum_serialized=pose.serialized
                )

    def _load_from_disk(self):
        # this will load everything except pose and pose_ts
        super()._load_from_disk()

        self._load_pose_and_ts_from_disk()

    def _load_pose_and_ts_from_disk(self):
        directory = self._camera_localizations_directory
        if not self._camera_localizer:
            return

        file_name = self._camera_localization_file_name(self._camera_localizer)
        pldata = fm.load_pldata_file(directory, file_name)
        self._camera_localizer.pose_bisector = pm.Bisector(
            pldata.data, pldata.timestamps
        )

    @property
    def item(self):
        return self._camera_localizer

    @property
    def items(self):
        return [self._camera_localizer] if self._camera_localizer else []

    @property
    def item_names(self):
        return [self._camera_localizer.name] if self._camera_localizer else []

    @property
    def _item_class(self):
        return CameraLocalizer

    @property
    def _storage_file_name(self):
        return "camera_localizers.msgpack"

    @property
    def _camera_localizations_directory(self):
        return os.path.join(self._storage_folder_path, "camera-poses")

    def _camera_localization_file_name(self, camera_localizer):
        file_name = camera_localizer.name + "-" + camera_localizer.unique_id
        return self.get_valid_filename(file_name)

    def _camera_localization_file_path(self, camera_localizer):
        return os.path.join(
            self._camera_localizations_directory,
            self._camera_localization_file_name(camera_localizer),
        )
