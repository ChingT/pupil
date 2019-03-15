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
        camera_intrinsics,
        task_manager,
        get_current_trim_mark_range,
        recording_uuid,
    ):
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._optimization_storage = optimization_storage
        self._marker_location_storage = marker_location_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._recording_uuid = recording_uuid

        self._task = None

    def calculate(self, optimization):
        def on_yield_optimization(result):
            optimization.result, camera_intrinsics = result
            self._model_storage.update_extrinsics_opt(optimization.result)
            if optimization.optimize_camera_intrinsics:
                self._camera_intrinsics.update_camera_matrix(camera_intrinsics.K)
                self._camera_intrinsics.update_dist_coefs(camera_intrinsics.D)

            optimization.status = "Optimization {:.0f}% complete".format(
                self._task.progress * 100
            )

        def on_completed_optimization(_):
            optimization.status = "Optimization successful"
            if optimization.optimize_camera_intrinsics:
                self._camera_intrinsics.save(self._optimization_storage._rec_dir)
            self._optimization_storage.save_to_disk()
            self.on_optimization_computed()

        if self._task is not None and self._task.running:
            self._task.kill(None)
        self._model_storage.reset()
        optimization.status = "Optimization 0% complete"
        self.on_optimization_calculating()
        self._task = worker.create_optimization.create_task(
            optimization, self._marker_location_storage
        )
        self._task.add_observer("on_yield", on_yield_optimization)
        self._task.add_observer("on_completed", on_completed_optimization)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task_manager.add_task(self._task)
        return self._task

    def on_optimization_calculating(self):
        pass

    def on_optimization_computed(self):
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
