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

import numpy as np

import make_unique
from head_pose_tracker import model, worker
from observable import Observable

logger = logging.getLogger(__name__)


class Markers3DModel(model.storage.StorageItem):
    version = 1

    def __init__(
        self,
        unique_id,
        name,
        recording_uuid,
        frame_index_range,
        origin_marker_id=None,
        status="Not calculated yet",
        result=None,
    ):
        self.unique_id = unique_id
        self.name = name
        self.recording_uuid = recording_uuid
        self.frame_index_range = frame_index_range
        self.origin_marker_id = origin_marker_id
        self.status = status

        if result is not None:
            self.result = {
                marker_id: np.array(extrinsics)
                for marker_id, extrinsics in result.items()
            }
            self.result_vis = {
                marker_id: worker.utils.convert_marker_extrinsics_to_points_3d(
                    extrinsics
                )
                for marker_id, extrinsics in result.items()
            }
        else:
            self.result = None
            self.result_vis = {}

        self.centroid = np.zeros((3,), dtype=np.float32)
        self.calculate_centroid()

        self.optimize_camera_intrinsics = False

    def reset(self):
        self.status = "Not calculated yet"
        self.result = None
        self.result_vis = {}
        self.origin_marker_id = None
        self.centroid = np.zeros((3,), dtype=np.float32)

    def update_result(self, markers_3d_model_result):
        self.result = markers_3d_model_result.result
        self.result_vis = markers_3d_model_result.result_vis
        self.origin_marker_id = markers_3d_model_result.origin_marker_id
        self.calculate_centroid()

    def calculate_centroid(self):
        try:
            self.centroid = np.mean(list(self.result_vis.values()), axis=(0, 1))
        except IndexError:
            self.centroid = np.zeros((3,), dtype=np.float32)

    @staticmethod
    def from_tuple(tuple_):
        return Markers3DModel(*tuple_)

    @property
    def as_tuple(self):
        if self.result:
            result = {key: value.tolist() for key, value in self.result.items()}
        else:
            result = {}
        return (
            self.unique_id,
            self.name,
            self.recording_uuid,
            self.frame_index_range,
            self.origin_marker_id,
            self.status,
            result,
        )


class Markers3DModelStorage(model.storage.Storage, Observable):
    _markers_3d_model_suffix = "plmodel"

    def __init__(self, rec_dir, plugin, get_recording_index_range, recording_uuid):
        super().__init__(plugin)
        self._rec_dir = rec_dir
        self._get_recording_index_range = get_recording_index_range
        self._recording_uuid = recording_uuid

        self._markers_3d_models = []
        self._load_from_disk()
        if not self._markers_3d_models:
            self._add_default_markers_3d_model()

    def _add_default_markers_3d_model(self):
        self.add(self.create_default_markers_3d_model())

    def create_default_markers_3d_model(self):
        return Markers3DModel(
            unique_id=Markers3DModel.create_new_unique_id(),
            name=make_unique.by_number_at_end(
                "Default Markers 3D Model", self.item_names
            ),
            recording_uuid=self._recording_uuid,
            frame_index_range=self._get_recording_index_range(),
        )

    def add(self, markers_3d_model):
        if any(
            c.unique_id == markers_3d_model.unique_id for c in self._markers_3d_models
        ):
            logger.warning(
                "Did not add markers_3d_model {} because it is already in the "
                "storage".format(markers_3d_model.name)
            )
            return
        self._markers_3d_models.append(markers_3d_model)
        self._markers_3d_models.sort(key=lambda c: c.name)

    def rename(self, markers_3d_model, new_name):
        old_markers_3d_model_file_path = self._markers_3d_model_file_path(
            markers_3d_model
        )
        markers_3d_model.name = new_name
        new_markers_3d_model_file_path = self._markers_3d_model_file_path(
            markers_3d_model
        )
        try:
            os.rename(old_markers_3d_model_file_path, new_markers_3d_model_file_path)
        except FileNotFoundError:
            pass

    def get_or_none(self):
        try:
            return next(c for c in self._markers_3d_models)
        except StopIteration:
            return None

    def _load_from_disk(self):
        try:
            markers_3d_model_files = [
                file_name
                for file_name in os.listdir(self._markers_3d_model_folder)
                if file_name.endswith(self._markers_3d_model_suffix)
            ]
        except FileNotFoundError:
            return

        if len(markers_3d_model_files) == 0:
            return
        elif len(markers_3d_model_files) > 1:
            logger.warning(
                "There should be only one markers_3d_model file in "
                "{}".format(self._markers_3d_model_folder)
            )
        self._load_markers_3d_model_from_file(markers_3d_model_files[0])

    def _load_markers_3d_model_from_file(self, file_name):
        file_path = os.path.join(self._markers_3d_model_folder, file_name)
        markers_3d_model_tuple = self._load_data_from_file(file_path)
        if markers_3d_model_tuple:
            markers_3d_model = Markers3DModel.from_tuple(markers_3d_model_tuple)
            if not self._from_same_recording(markers_3d_model):
                # the index range from another recording is useless and can lead
                # to confusion if it is rendered somewhere
                markers_3d_model.frame_index_range = [0, 0]
            self.add(markers_3d_model)

    def save_to_disk(self):
        os.makedirs(self._markers_3d_model_folder, exist_ok=True)
        markers_3d_models_from_same_recording = (
            opt for opt in self._markers_3d_models if self._from_same_recording(opt)
        )
        for markers_3d_model in markers_3d_models_from_same_recording:
            self._save_data_to_file(
                self._markers_3d_model_file_path(markers_3d_model),
                markers_3d_model.as_tuple,
            )

    def _from_same_recording(self, markers_3d_model):
        # There is a very similar, but public method in the Markers3DModelController.
        # This method only exists because its extremely inconvenient to access
        # controllers from storages and the logic is very simple.
        return markers_3d_model.recording_uuid == self._recording_uuid

    @property
    def items(self):
        return self._markers_3d_models

    @property
    def item_names(self):
        return [opt.name for opt in self._markers_3d_models]

    @property
    def _item_class(self):
        return Markers3DModel

    @property
    def _markers_3d_model_folder(self):
        return os.path.join(self._rec_dir, "markers_3d_model")

    def _markers_3d_model_file_name(self, markers_3d_model):
        file_name = "{}-{}.{}".format(
            markers_3d_model.name,
            markers_3d_model.unique_id,
            self._markers_3d_model_suffix,
        )
        return self.get_valid_filename(file_name)

    def _markers_3d_model_file_path(self, markers_3d_model):
        return os.path.join(
            self._markers_3d_model_folder,
            self._markers_3d_model_file_name(markers_3d_model),
        )
