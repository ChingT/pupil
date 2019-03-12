"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import tasklib
from head_pose_tracker import worker
from observable import Observable


class OptimizationController(Observable):
    def __init__(
        self,
        controller_storage,
        model_storage,
        optimization_storage,
        marker_location_storage,
        task_manager,
        get_current_trim_mark_range,
        recording_uuid,
    ):
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._optimization_storage = optimization_storage
        self._marker_location_storage = marker_location_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._recording_uuid = recording_uuid

    def calculate(self, optimization):
        def on_optimization_yield(result):
            optimization.result = result
            self._optimization_storage.save_to_disk()
            self._model_storage.update_extrinsics_opt(optimization.result)

        task = worker.create_optimization.create_task(
            self._controller_storage,
            self._model_storage,
            all_marker_locations=self._marker_location_storage,
        )
        task.add_observer("on_yield", on_optimization_yield)
        task.add_observer("on_completed", self.on_optimization_computed)
        task.add_observer("on_exception", tasklib.raise_exception)
        self._task_manager.add_task(task)
        return task

    def on_optimization_computed(self, optimization):
        pass

    def set_optimization_range_from_current_trim_marks(self, optimization):
        optimization.frame_index_range = self._get_current_trim_mark_range()

    def is_from_same_recording(self, optimization):
        """
        False if the optimization file was copied from another recording directory
        """
        return (
            optimization is not None
            and optimization.recording_uuid == self._recording_uuid
        )
