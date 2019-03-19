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
import os

import file_methods as fm
import make_unique
import player_methods as pm
from head_pose_tracker import model
from observable import Observable

logger = logging.getLogger(__name__)


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


class CameraLocalizerStorage(model.SingleFileStorage, Observable):
    def __init__(self, rec_dir, plugin, get_recording_index_range):
        super().__init__(rec_dir, plugin)
        self._get_recording_index_range = get_recording_index_range

        self._camera_localizers = []
        self._load_from_disk()
        if not self._camera_localizers:
            self._add_default_camera_localizer()

    def _add_default_camera_localizer(self):
        self.add(self.create_default_camera_localizer())

    def create_default_camera_localizer(self):
        return CameraLocalizer(
            unique_id=CameraLocalizer.create_new_unique_id(),
            name=make_unique.by_number_at_end(
                "Default Camera Localizer", self.item_names
            ),
            localization_index_range=self._get_recording_index_range(),
        )

    def add(self, camera_localizer):
        self._camera_localizers.append(camera_localizer)
        self._camera_localizers.sort(key=lambda g: g.name)

    def rename(self, camera_localizer, new_name):
        old_localization_file_path = self._camera_localization_file_path(
            camera_localizer
        )
        camera_localizer.name = new_name
        new_localization_file_path = self._camera_localization_file_path(
            camera_localizer
        )
        self._rename_localization_file(
            old_localization_file_path, new_localization_file_path
        )

    def _rename_localization_file(
        self, old_localization_file_path, new_localization_file_path
    ):
        try:
            os.rename(
                old_localization_file_path + ".pldata",
                new_localization_file_path + ".pldata",
            )
            os.rename(
                old_localization_file_path + "_timestamps.npy",
                new_localization_file_path + "_timestamps.npy",
            )
        except FileNotFoundError:
            pass

    def save_to_disk(self):
        # this will save everything except pose and pose_ts
        super().save_to_disk()

        self._save_pose_and_ts_to_disk()

    def _save_pose_and_ts_to_disk(self):
        directory = self._camera_localizations_directory
        os.makedirs(directory, exist_ok=True)
        for camera_localizer in self._camera_localizers:
            file_name = self._camera_localization_file_name(camera_localizer)
            with fm.PLData_Writer(directory, file_name) as writer:
                for pose_ts, pose in zip(
                    camera_localizer.pose_bisector.timestamps,
                    camera_localizer.pose_bisector.data,
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
        for camera_localizer in self._camera_localizers:
            file_name = self._camera_localization_file_name(camera_localizer)
            pldata = fm.load_pldata_file(directory, file_name)
            camera_localizer.pose_bisector = pm.Bisector(pldata.data, pldata.timestamps)

    @property
    def _storage_file_name(self):
        return "camera_localizers.msgpack"

    @property
    def _item_class(self):
        return CameraLocalizer

    @property
    def items(self):
        return self._camera_localizers

    @property
    def item_names(self):
        return [camera_localizer.name for camera_localizer in self._camera_localizers]

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

    def get_or_none(self):
        try:
            return next(c for c in self._camera_localizers)
        except StopIteration:
            return None
