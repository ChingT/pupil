"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import copy
import logging
import os

import file_methods as fm
import make_unique
from head_pose_tracker import model
from observable import Observable

logger = logging.getLogger(__name__)


class OptimizationStorage(model.storage.Storage, Observable):
    _optimization_suffix = "plcal"

    def __init__(
        self, rec_dir, plugin, get_recording_index_range, recording_uuid, model_storage
    ):
        super().__init__(plugin)
        self._rec_dir = rec_dir
        self._get_recording_index_range = get_recording_index_range
        self._recording_uuid = recording_uuid
        self._model_storage = model_storage

        self._optimizations = []
        self._load_from_disk()
        if not self._optimizations:
            self._add_default_optimization()

    def _add_default_optimization(self):
        self.add(self.create_default_optimization())

    def create_default_optimization(self):
        return model.Optimization(
            unique_id=model.Optimization.create_new_unique_id(),
            name=make_unique.by_number_at_end("Default Optimization", self.item_names),
            recording_uuid=self._recording_uuid,
            mapping_method="3d",
            frame_index_range=self._get_recording_index_range(),
            minimum_confidence=0.8,
        )

    def duplicate_optimization(self, optimization):
        new_optimization = copy.deepcopy(optimization)
        new_optimization.name = make_unique.by_number_at_end(
            new_optimization.name + " Copy", self.item_names
        )
        new_optimization.unique_id = model.Optimization.create_new_unique_id()
        return new_optimization

    def add(self, optimization):
        if any(c.unique_id == optimization.unique_id for c in self._optimizations):
            logger.warning(
                "Did not add optimization {} because it is already in the "
                "storage".format(optimization.name)
            )
            return
        self._optimizations.append(optimization)
        self._optimizations.sort(key=lambda c: c.name)

        if optimization.result:
            self._model_storage.update_extrinsics_opt(optimization.result)

    def delete(self, optimization):
        self._optimizations.remove(optimization)
        self._delete_optimization_file(optimization)

    def _delete_optimization_file(self, optimization):
        try:
            os.remove(self._optimization_file_path(optimization))
        except FileNotFoundError:
            pass

    def rename(self, optimization, new_name):
        old_optimization_file_path = self._optimization_file_path(optimization)
        optimization.name = new_name
        new_optimization_file_path = self._optimization_file_path(optimization)
        try:
            os.rename(old_optimization_file_path, new_optimization_file_path)
        except FileNotFoundError:
            pass

    def get_first_or_none(self):
        if self._optimizations:
            return self._optimizations[0]
        else:
            return None

    def get_or_none(self, unique_id):
        try:
            return next(c for c in self._optimizations if c.unique_id == unique_id)
        except StopIteration:
            return None

    def _load_from_disk(self):
        try:
            # we sort because listdir sometimes returns files in weird order
            for file_name in sorted(os.listdir(self._optimization_folder)):
                if file_name.endswith(self._optimization_suffix):
                    self._load_optimization_from_file(file_name)
        except FileNotFoundError:
            pass
        self._load_recorded_optimizations()

    def _load_optimization_from_file(self, file_name):
        file_path = os.path.join(self._optimization_folder, file_name)
        optimization_tuple = self._load_data_from_file(file_path)
        if optimization_tuple:
            optimization = model.Optimization.from_tuple(optimization_tuple)
            if not self._from_same_recording(optimization):
                # the index range from another recording is useless and can lead
                # to confusion if it is rendered somewhere
                optimization.frame_index_range = [0, 0]
            self.add(optimization)

    def _load_recorded_optimizations(self):
        notifications = fm.load_pldata_file(self._rec_dir, "notify")
        for topic, data in zip(notifications.topics, notifications.data):
            if topic == "notify.optimization.optimization_data":
                try:
                    opt_result = model.OptimizationResult(
                        mapping_plugin_name=data["mapper_name"],
                        mapper_args=data["mapper_args"],
                    )
                except KeyError:
                    # notifications from old recordings will not have these fields!
                    continue
                mapping_method = "2d" if "2d" in data["optimization_method"] else "3d"
                # the unique id needs to be the same at every start or otherwise the
                # same optimizations would be added again and again. The timestamp is
                # the easiest datum that differs between optimizations but is the same
                # for every start
                unique_id = model.Optimization.create_unique_id_from_string(
                    str(data["timestamp"])
                )
                optimization = model.Optimization(
                    unique_id=unique_id,
                    name=make_unique.by_number_at_end(
                        "Recorded Optimization", self.item_names
                    ),
                    recording_uuid=self._recording_uuid,
                    mapping_method=mapping_method,
                    frame_index_range=self._get_recording_index_range(),
                    minimum_confidence=0.8,
                    is_offline_optimization=False,
                    result=opt_result,
                )
                self.add(optimization)

    def save_to_disk(self):
        os.makedirs(self._optimization_folder, exist_ok=True)
        optimizations_from_same_recording = (
            opt for opt in self._optimizations if self._from_same_recording(opt)
        )
        for optimization in optimizations_from_same_recording:
            self._save_data_to_file(
                self._optimization_file_path(optimization), optimization.as_tuple
            )

    def _from_same_recording(self, optimization):
        # There is a very similar, but public method in the OptimizationController.
        # This method only exists because its extremely inconvenient to access
        # controllers from storages and the logic is very simple.
        return optimization.recording_uuid == self._recording_uuid

    @property
    def items(self):
        return self._optimizations

    @property
    def item_names(self):
        return [opt.name for opt in self._optimizations]

    @property
    def _item_class(self):
        return model.Optimization

    @property
    def _optimization_folder(self):
        return os.path.join(self._rec_dir, "marker_3d_model")

    def _optimization_file_name(self, optimization):
        file_name = "{}-{}.{}".format(
            optimization.name, optimization.unique_id, self._optimization_suffix
        )
        return self.get_valid_filename(file_name)

    def _optimization_file_path(self, optimization):
        return os.path.join(
            self._optimization_folder, self._optimization_file_name(optimization)
        )
