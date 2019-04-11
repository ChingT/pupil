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
        general_settings,
        marker_location_storage,
        markers_3d_model_storage,
        camera_intrinsics,
        task_manager,
        get_current_trim_mark_range,
        all_timestamps,
        rec_dir,
    ):
        self._general_settings = general_settings
        self._marker_location_storage = marker_location_storage
        self._markers_3d_model_storage = markers_3d_model_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._all_timestamps = all_timestamps
        self._rec_dir = rec_dir

        self._task = None

        marker_location_controller.add_observer(
            "on_marker_detection_ended", self._on_marker_detection_ended
        )

    def _on_marker_detection_ended(self):
        if (
            self._markers_3d_model_storage.is_from_same_recording
            and not self._markers_3d_model_storage.calculated
        ):
            self.calculate()
        else:
            self.on_markers_3d_model_optimization_had_completed_before()

    def calculate(self):
        self._reset()
        self._create_optimize_markers_3d_model_task()

    def _reset(self):
        if self._task is not None and self._task.running:
            self._task.kill(None)

        self._general_settings.markers_3d_model_status = "Not calculated yet"
        self._markers_3d_model_storage.result = None

    def _create_optimize_markers_3d_model_task(self):
        def on_yield(result):
            self._update_result(result)
            self._general_settings.markers_3d_model_status = "{:.0f}% completed".format(
                self._task.progress * 100
            )

        def on_completed(_):
            if self._markers_3d_model_storage.calculated:
                self._general_settings.markers_3d_model_status = "successful"
                self._camera_intrinsics.save(self._rec_dir)
                logger.info(
                    "markers 3d model '{}' optimization completed".format(
                        self._markers_3d_model_storage.name
                    )
                )
                self.on_markers_3d_model_optimization_completed()
            else:
                self._general_settings.markers_3d_model_status = "failed"
                logger.info(
                    "markers 3d model '{}' optimization failed".format(
                        self._markers_3d_model_storage.name
                    )
                )

            self._markers_3d_model_storage.save_plmodel_to_disk()

        self._task = worker.optimize_markers_3d_model.create_task(
            self._all_timestamps, self._marker_location_storage, self._general_settings
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
                self._markers_3d_model_storage.name
            )
        )
        self._general_settings.markers_3d_model_status = "0% completed"

    def _update_result(self, result):
        model_data, intrinsics = result
        self._markers_3d_model_storage.result = model_data
        self._camera_intrinsics.update_camera_matrix(intrinsics["camera_matrix"])
        self._camera_intrinsics.update_dist_coefs(intrinsics["dist_coefs"])

    def on_markers_3d_model_optimization_had_completed_before(self):
        pass

    def on_markers_3d_model_optimization_started(self):
        pass

    def on_markers_3d_model_optimization_completed(self):
        pass

    def set_range_from_current_trim_marks(self):
        self._general_settings.markers_3d_model_frame_index_range = (
            self._get_current_trim_mark_range()
        )
