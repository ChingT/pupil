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
from camera_extrinsics_measurer import worker, camera_names
from observable import Observable

logger = logging.getLogger(__name__)


class OfflineLocalizationController(Observable):
    def __init__(
        self,
        detection_controller,
        optimization_controller,
        general_settings,
        detection_storage,
        optimization_storage,
        localization_storage,
        camera_intrinsics_dict,
        task_manager,
        current_trim_mark_ts_range,
        all_timestamps_dict,
    ):
        self._detection_controller = detection_controller
        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._optimization_storage = optimization_storage
        self._localization_storage = localization_storage
        self._camera_intrinsics_dict = camera_intrinsics_dict
        self._task_manager = task_manager
        self._current_trim_mark_ts_range = current_trim_mark_ts_range
        self._all_timestamps_dict = all_timestamps_dict
        self._task = None

        if self._localization_storage.calculated:
            self.status = "calculated"
        else:
            self.status = self._default_status

        optimization_controller.add_observer(
            "on_optimization_had_completed_before",
            self._on_optimization_had_completed_before,
        )
        optimization_controller.add_observer(
            "on_optimization_started", self._on_optimization_started
        )
        optimization_controller.add_observer(
            "on_optimization_completed", self._on_optimization_completed
        )

    @property
    def _default_status(self):
        return "Not calculated yet"

    def _on_optimization_had_completed_before(self, camera_name):
        self.calculate(camera_name)

    def _on_optimization_started(self, camera_name):
        self.reset(camera_name)

    def _on_optimization_completed(self, camera_name):
        self.calculate(camera_name)

    def calculate(self, camera_name):
        if not self._check_valid_markers_3d_model():
            self.on_localization_ended()
            return

        self.reset(camera_name)
        self._create_localization_task(camera_name)

    def _check_valid_markers_3d_model(self):
        if not self._optimization_storage.calculated:
            error_message = (
                "failed: markers 3d model '{}' should be calculated before calculating"
                " camera localization".format(self._optimization_storage.name)
            )
            self._abort_calculation(error_message)
            return False
        return True

    def _abort_calculation(self, error_message):
        logger.error(error_message)
        self.status = error_message
        self.on_localization_could_not_be_started()

    def reset(self, camera_name):
        self.cancel_task()
        self._localization_storage.set_to_default_values(camera_name)
        self.status = self._default_status

    def _create_localization_task(self, camera_name):
        def on_yield(data_pairs):
            self._insert_pose_bisector(camera_name, data_pairs)
            self.status = "{:.0f}% completed".format(self._task.progress * 100)

        def on_completed(_):
            self.status = "successfully completed"
            logger.info("[{}] camera localization completed".format(camera_name))
            self.on_localization_ended()
            self._on_routine_ended(camera_name)

        def on_canceled_or_killed():
            logger.info("[{}] camera localization canceled".format(camera_name))
            self.on_localization_ended()

        self._task = self._create_task(camera_name)
        self._task.add_observer("on_yield", on_yield)
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_canceled_or_killed", on_canceled_or_killed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer("on_started", self.on_localization_started)
        logger.info("[{}] Start camera localization".format(camera_name))
        self.status = "0% completed"

    def _create_task(self, camera_name):
        args = (
            self._all_timestamps_dict[camera_name],
            self._detection_storage.markers_bisector[camera_name],
            self._detection_storage.frame_index_to_num_markers[camera_name],
            self._optimization_storage.marker_id_to_extrinsics,
            self._camera_intrinsics_dict[camera_name],
        )
        return self._task_manager.create_background_task(
            name="camera localization",
            routine_or_generator_function=worker.offline_localization,
            pass_shared_memory=True,
            args=args,
        )

    def _insert_pose_bisector(self, camera_name, data_pairs):
        for timestamp, pose in data_pairs:
            self._localization_storage.pose_bisector[camera_name].insert(
                timestamp, pose
            )
        self.on_localization_yield()

    def cancel_task(self):
        if self.is_running_task:
            self._task.kill(None)

    @property
    def is_running_task(self):
        return self._task is not None and self._task.running

    @property
    def progress(self):
        return self._task.progress if self.is_running_task else 0.0

    def set_range_from_current_trim_marks(self):
        self._general_settings.localization_frame_ts_range = (
            self._current_trim_mark_ts_range()
        )

    def on_localization_could_not_be_started(self):
        pass

    def on_localization_started(self):
        pass

    def on_localization_yield(self):
        pass

    def on_localization_ended(self):
        pass

    def _on_routine_ended(self, camera_name):
        index = camera_names.index(camera_name)
        try:
            self._detection_controller.calculate(camera_names[index + 1])
        except IndexError:
            self._convert_to_world_coordinate()

    def _convert_to_world_coordinate(self):
        logger.info("Start converting to world coordinate")
        pose_bisector_converted = worker.convert_to_world_coordinate(
            self._all_timestamps_dict["world"], self._localization_storage.pose_bisector
        )
        self._localization_storage.pose_bisector_converted = pose_bisector_converted
        logger.info("Converting to world coordinate completed")
        self._localization_storage.save_pldata_to_disk()
