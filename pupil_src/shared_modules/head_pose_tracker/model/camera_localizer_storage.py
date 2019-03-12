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
from head_pose_tracker import model
from observable import Observable

logger = logging.getLogger(__name__)


class CameraLocalizerStorage(model.SingleFileStorage, Observable):
    def __init__(
        self, optimization_storage, rec_dir, plugin, get_recording_index_range
    ):
        super().__init__(rec_dir, plugin)
        self._optimization_storage = optimization_storage
        self._get_recording_index_range = get_recording_index_range
        self._camera_localizers = []
        self._load_from_disk()
        if not self._camera_localizers:
            self._add_default_camera_localizer()

    def _add_default_camera_localizer(self):
        self.add(self.create_default_camera_localizer())

    def create_default_camera_localizer(self):
        default_optimization = self._optimization_storage.get_first_or_none()
        if default_optimization:
            optimization_unique_id = default_optimization.unique_id
        else:
            optimization_unique_id = ""
        return model.CameraLocalizer(
            unique_id=model.CameraLocalizer.create_new_unique_id(),
            name=make_unique.by_number_at_end("Camera Localizer", self.item_names),
            optimization_unique_id=optimization_unique_id,
            mapping_index_range=self._get_recording_index_range(),
            validation_index_range=self._get_recording_index_range(),
            validation_outlier_threshold_deg=5.0,
        )

    def duplicate_camera_localizer(self, camera_localizer):
        return model.CameraLocalizer(
            unique_id=camera_localizer.create_new_unique_id(),
            name=make_unique.by_number_at_end(
                camera_localizer.name + " Copy", self.item_names
            ),
            optimization_unique_id=camera_localizer.optimization_unique_id,
            mapping_index_range=camera_localizer.mapping_index_range,
            validation_index_range=camera_localizer.validation_index_range,
            validation_outlier_threshold_deg=camera_localizer.validation_outlier_threshold_deg,
            manual_correction_x=camera_localizer.manual_correction_x,
            manual_correction_y=camera_localizer.manual_correction_y,
            activate_gaze=camera_localizer.activate_gaze,
            # We cannot deep copy gaze, so we don't.
            # All others left at their default.
        )

    def add(self, camera_localizer):
        self._camera_localizers.append(camera_localizer)
        self._camera_localizers.sort(key=lambda g: g.name)

    def delete(self, camera_localizer):
        self._camera_localizers.remove(camera_localizer)
        self._delete_mapping_file(camera_localizer)

    def _delete_mapping_file(self, camera_localizer):
        mapping_file_path = self._camera_localization_file_path(camera_localizer)
        try:
            os.remove(mapping_file_path + ".pldata")
            os.remove(mapping_file_path + "_timestamps.npy")
        except FileNotFoundError:
            pass

    def rename(self, camera_localizer, new_name):
        old_mapping_file_path = self._camera_localization_file_path(camera_localizer)
        camera_localizer.name = new_name
        new_mapping_file_path = self._camera_localization_file_path(camera_localizer)
        self._rename_mapping_file(old_mapping_file_path, new_mapping_file_path)

    def _rename_mapping_file(self, old_mapping_file_path, new_mapping_file_path):
        try:
            os.rename(
                old_mapping_file_path + ".pldata", new_mapping_file_path + ".pldata"
            )
            os.rename(
                old_mapping_file_path + "_timestamps.npy",
                new_mapping_file_path + "_timestamps.npy",
            )
        except FileNotFoundError:
            pass

    def save_to_disk(self):
        # this will save everything except gaze and gaze_ts
        super().save_to_disk()

        self._save_gaze_and_ts_to_disk()

    def _save_gaze_and_ts_to_disk(self):
        directory = self._camera_localizations_directory
        os.makedirs(directory, exist_ok=True)
        for camera_localizer in self._camera_localizers:
            file_name = self._camera_localization_file_name(camera_localizer)
            with fm.PLData_Writer(directory, file_name) as writer:
                for gaze_ts, gaze in zip(
                    camera_localizer.gaze_ts, camera_localizer.gaze
                ):
                    writer.append_serialized(
                        gaze_ts, topic="gaze", datum_serialized=gaze.serialized
                    )

    def _load_from_disk(self):
        # this will load everything except gaze and gaze_ts
        super()._load_from_disk()

        self._load_gaze_and_ts_from_disk()

    def _load_gaze_and_ts_from_disk(self):
        directory = self._camera_localizations_directory
        for camera_localizer in self._camera_localizers:
            file_name = self._camera_localization_file_name(camera_localizer)
            pldata = fm.load_pldata_file(directory, file_name)
            camera_localizer.gaze = pldata.data
            camera_localizer.gaze_ts = pldata.timestamps

    @property
    def _storage_file_name(self):
        return "camera_localizers.msgpack"

    @property
    def _item_class(self):
        return model.CameraLocalizer

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
