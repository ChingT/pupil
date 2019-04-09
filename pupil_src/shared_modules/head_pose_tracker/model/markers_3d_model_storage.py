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

from head_pose_tracker import model
from observable import Observable

logger = logging.getLogger(__name__)


class Markers3DModel(model.StorageItem):
    version = 1

    def __init__(
        self,
        name,
        recording_uuid,
        frame_index_range,
        status="Not calculated yet",
        result=None,
        user_defined_origin_marker_id=None,
    ):
        self.name = name
        self.recording_uuid = recording_uuid
        self.frame_index_range = tuple(frame_index_range)
        self.status = status
        self.result = result
        self.user_defined_origin_marker_id = user_defined_origin_marker_id

        self.optimize_camera_intrinsics = False
        self.show_marker_id = False

    @staticmethod
    def from_tuple(tuple_):
        return Markers3DModel(*tuple_)

    @property
    def as_tuple(self):
        return (
            self.name,
            self.recording_uuid,
            self.frame_index_range,
            self.status,
            self.result,
            self.user_defined_origin_marker_id,
        )

    @property
    def calculated(self):
        return self.result and self.result["marker_id_to_extrinsics"]


class Markers3DModelStorage(model.Storage, Observable):
    _markers_3d_model_suffix = "plmodel"

    def __init__(self, rec_dir, plugin, get_recording_index_range, recording_uuid):
        self._recording_uuid = recording_uuid
        super().__init__(rec_dir, plugin, get_recording_index_range)

    def _find_file_path(self):
        try:
            markers_3d_model_files = [
                file_name
                for file_name in os.listdir(self._storage_folder_path)
                if file_name.endswith(self._markers_3d_model_suffix)
            ]
        except FileNotFoundError:
            return

        if len(markers_3d_model_files) == 0:
            return
        elif len(markers_3d_model_files) > 1:
            logger.warning(
                "There should be only one markers_3d_model file in "
                "{}".format(self._storage_folder_path)
            )
        file_name = markers_3d_model_files[0]
        file_path = os.path.join(self._storage_folder_path, file_name)
        return file_path

    def save_to_disk(self):
        if self.is_from_same_recording:
            super().save_to_disk()

    def _create_default_item(self):
        return Markers3DModel(
            name="Default",
            recording_uuid=self._recording_uuid,
            frame_index_range=self._get_recording_index_range(),
        )

    @property
    def _item_class(self):
        return Markers3DModel

    @property
    def _storage_folder_path(self):
        return os.path.join(self._rec_dir, "Markers 3D Model")

    @property
    def _storage_file_name(self):
        file_name = "{}.{}".format(self.item.name, self._markers_3d_model_suffix)
        return self._get_valid_filename(file_name)

    def rename(self, new_name):
        old_markers_3d_model_file_path = self._storage_file_path
        self.item.name = new_name
        new_markers_3d_model_file_path = self._storage_file_path
        try:
            os.rename(old_markers_3d_model_file_path, new_markers_3d_model_file_path)
        except FileNotFoundError:
            pass

    @property
    def is_from_same_recording(self):
        # There is a very similar, but public method in the Markers3DModelController.
        # This method only exists because its extremely inconvenient to access
        # controllers from storages and the logic is very simple.
        return self.item.recording_uuid == self._recording_uuid
