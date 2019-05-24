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

import numpy as np

import tasklib
from camera_extrinsics_measurer import worker
from observable import Observable

logger = logging.getLogger(__name__)


class OfflineOptimizationController(Observable):
    def __init__(
        self,
        detection_controller,
        general_settings,
        detection_storage,
        optimization_storage,
        camera_intrinsics_dict,
        task_manager,
        current_trim_mark_ts_range,
        all_timestamps_dict,
        rec_dir,
    ):
        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._optimization_storage = optimization_storage
        self._camera_intrinsics_dict = camera_intrinsics_dict
        self._task_manager = task_manager
        self._current_trim_mark_ts_range = current_trim_mark_ts_range
        self._all_timestamps_dict = all_timestamps_dict
        self._rec_dir = rec_dir

        self._task = None

        if self._optimization_storage.calculated:
            self.status = "calculated"
        else:
            self.status = self._default_status

        detection_controller.add_observer(
            "on_detection_ended", self._on_detection_ended
        )

    @property
    def _default_status(self):
        return "Not calculated yet"

    def _on_detection_ended(self, camera_name):
        self.calculate(camera_name)

    def calculate(self, camera_name):
        if self._general_settings.optimize_camera_intrinsics:
            self._reset()
            self._create_optimization_task(camera_name)
        else:
            self.on_optimization_had_completed_before(camera_name)

    def _reset(self):
        self.cancel_task()
        # self._optimization_storage.set_to_default_values()
        self.status = self._default_status

    def _create_optimization_task(self, camera_name):
        def on_started():
            self.on_optimization_started(camera_name)

        def on_yield(result):
            self._update_result(camera_name, result)
            self.status = "{:.0f}% completed".format(self._task.progress * 100)

        def on_completed(_):
            if self._optimization_storage.calculated:
                self._camera_intrinsics_dict[camera_name].save(self._rec_dir)
                self.status = "successfully completed"
                self.on_optimization_completed(camera_name)
            else:
                if self._general_settings.user_defined_origin_marker_id is not None:
                    reason = (
                        "not enough markers with the defined origin marker id "
                        "were collected"
                    )
                else:
                    reason = "not enough markers were collected"

                self.status = "failed: " + reason
            logger.info(
                "[{}] markers 3d model optimization '{}' ".format(
                    self.status, camera_name
                )
            )

        self._task = self._create_task(camera_name)
        self._task.add_observer("on_yield", on_yield)
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer("on_started", on_started)
        logger.info("[{}] Start markers 3d model optimization".format(camera_name))
        self.status = "0% completed"

    def _create_task(self, camera_name):
        args = (
            camera_name,
            self._all_timestamps_dict[camera_name],
            self._general_settings.user_defined_origin_marker_id,
            self._optimization_storage.marker_id_to_extrinsics,
            self._general_settings.optimize_camera_intrinsics,
            self._detection_storage.markers_bisector[camera_name],
            self._detection_storage.frame_index_to_num_markers[camera_name],
            self._camera_intrinsics_dict[camera_name],
            self._rec_dir,
            self._general_settings.debug,
        )
        return self._task_manager.create_background_task(
            name="markers 3d model optimization",
            routine_or_generator_function=worker.offline_optimization,
            pass_shared_memory=True,
            args=args,
        )

    def _update_result(self, camera_name, result):
        model_tuple, intrinsics_tuple = result
        self._optimization_storage.update_model(*model_tuple)
        self._camera_intrinsics_dict[camera_name].update_camera_matrix(
            intrinsics_tuple.camera_matrix
        )
        self._camera_intrinsics_dict[camera_name].update_dist_coefs(
            intrinsics_tuple.dist_coefs
        )
        print(np.around(self._camera_intrinsics_dict[camera_name].K, 8).tolist())
        print(np.around(self._camera_intrinsics_dict[camera_name].D, 8).tolist())

    def cancel_task(self):
        if self.is_running_task:
            self._task.kill(None)

    @property
    def is_running_task(self):
        return self._task is not None and self._task.running

    def set_range_from_current_trim_marks(self):
        self._general_settings.optimization_frame_ts_range = (
            self._current_trim_mark_ts_range()
        )

    def on_optimization_had_completed_before(self, camera_name):
        pass

    def on_optimization_started(self, camera_name):
        pass

    def on_optimization_completed(self, camera_name):
        pass
