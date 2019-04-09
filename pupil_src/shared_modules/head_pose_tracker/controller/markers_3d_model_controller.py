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

import tasklib
from head_pose_tracker import worker
from observable import Observable

logger = logging.getLogger(__name__)


class Markers3DModelController(Observable):
    def __init__(
        self,
        marker_location_controller,
        marker_location_storage,
        markers_3d_model_storage,
        camera_intrinsics,
        task_manager,
        get_current_trim_mark_range,
        all_timestamps,
        recording_uuid,
        rec_dir,
    ):
        self._markers_3d_model_storage = markers_3d_model_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._all_timestamps = all_timestamps
        self._recording_uuid = recording_uuid
        self._rec_dir = rec_dir

        self._marker_locations = marker_location_storage.item
        self._markers_3d_model = markers_3d_model_storage.item

        self._task = None

        marker_location_controller.add_observer(
            "on_marker_detection_ended", self._on_marker_detection_ended
        )

    def _on_marker_detection_ended(self):
        self.calculate(check_complete=True)

    def calculate(self, check_complete=False):
        if check_complete and self._markers_3d_model.calculated:
            self.on_markers_3d_model_optimization_had_completed_before()
        else:
            self._reset()
            self._create_optimize_markers_3d_model_task()

    def _reset(self):
        if self._task is not None and self._task.running:
            self._task.kill(None)

        self._markers_3d_model.status = "Not calculated yet"
        self._markers_3d_model.result = None

    def _create_optimize_markers_3d_model_task(self):
        def on_yield(result):
            self._update_result(result)
            self._markers_3d_model.status = "{:.0f}% completed".format(
                self._task.progress * 100
            )

        def on_completed(_):
            if self._markers_3d_model.calculated:
                self._markers_3d_model.status = "successful"
                self._camera_intrinsics.save(self._rec_dir)
                logger.info(
                    "markers 3d model '{}' optimization completed".format(
                        self._markers_3d_model.name
                    )
                )
                self.on_markers_3d_model_optimization_completed()
            else:
                self._markers_3d_model.status = "failed"
                logger.info(
                    "markers 3d model '{}' optimization failed".format(
                        self._markers_3d_model.name
                    )
                )

            self._markers_3d_model_storage.save_to_disk()

        self._task = worker.optimize_markers_3d_model.create_task(
            self._all_timestamps, self._marker_locations, self._markers_3d_model
        )
        self._task.add_observer("on_yield", on_yield)
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer(
            "on_started", self.on_markers_3d_model_optimization_started
        )
        self._task_manager.add_task(self._task)
        logger.info(
            "Start markers 3d model '{}' optimization".format(
                self._markers_3d_model.name
            )
        )
        self._markers_3d_model.status = "0% completed"

    def _update_result(self, result):
        model_data, intrinsics = result
        self._markers_3d_model.result = model_data
        self._camera_intrinsics.update_camera_matrix(intrinsics["camera_matrix"])
        self._camera_intrinsics.update_dist_coefs(intrinsics["dist_coefs"])

    def on_markers_3d_model_optimization_had_completed_before(self):
        pass

    def on_markers_3d_model_optimization_started(self):
        pass

    def on_markers_3d_model_optimization_completed(self):
        pass

    def set_range_from_current_trim_marks(self):
        self._markers_3d_model.frame_index_range = self._get_current_trim_mark_range()
