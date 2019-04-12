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

import player_methods as pm
import tasklib
from head_pose_tracker import worker
from observable import Observable

logger = logging.getLogger(__name__)


class CameraLocalizerController(Observable):
    def __init__(
        self,
        markers_3d_model_controller,
        general_settings,
        marker_location_storage,
        markers_3d_model_storage,
        camera_localizer_storage,
        task_manager,
        get_current_trim_mark_range,
        all_timestamps,
    ):
        self._general_settings = general_settings
        self._marker_location_storage = marker_location_storage
        self._markers_3d_model_storage = markers_3d_model_storage
        self._camera_localizer_storage = camera_localizer_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._all_timestamps = all_timestamps

        self._task = None

        if self._camera_localizer_storage.calculated:
            self.status = "calculated"
        else:
            self.status = self.default_status

        markers_3d_model_controller.add_observer(
            "on_markers_3d_model_optimization_had_completed_before",
            self._on_markers_3d_model_optimization_had_completed_before,
        )
        markers_3d_model_controller.add_observer(
            "on_markers_3d_model_optimization_started",
            self._on_markers_3d_model_optimization_started,
        )
        markers_3d_model_controller.add_observer(
            "on_markers_3d_model_optimization_completed",
            self._on_markers_3d_model_optimization_completed,
        )

    @property
    def default_status(self):
        return "Not calculated yet"

    def _on_markers_3d_model_optimization_had_completed_before(self):
        if not self._camera_localizer_storage.calculated:
            self.calculate()

    def _on_markers_3d_model_optimization_started(self):
        self.reset()

    def _on_markers_3d_model_optimization_completed(self):
        self.calculate()

    def calculate(self):
        if not self._check_valid_markers_3d_model():
            return

        self.reset()
        self._create_localization_task()

    def _check_valid_markers_3d_model(self):
        if not self._markers_3d_model_storage.calculated:
            error_message = (
                "You first need to calculate markers 3d model '{}' before calculating "
                "the camera localizer".format(self._markers_3d_model_storage.name)
            )
            self._abort_calculation(error_message)
            return False
        return True

    def _abort_calculation(self, error_message):
        logger.error(error_message)
        self.status = error_message
        self.on_calculation_could_not_be_started()
        # the pose from this localizer got cleared, so don't show it anymore

    def reset(self):
        self.cancel_task()
        self._camera_localizer_storage.pose_bisector = pm.Mutable_Bisector()
        self.status = self.default_status

    def _create_localization_task(self):
        def on_yield(data_pairs):
            self._insert_pose_bisector(data_pairs)
            self.status = "{:.0f}% completed".format(self._task.progress * 100)

        def on_completed(_):
            self.status = "successful"
            self._camera_localizer_storage.save_pldata_to_disk()
            logger.info("camera localization completed")
            self.on_camera_localization_ended()

        def on_canceled_or_killed():
            self._camera_localizer_storage.save_pldata_to_disk()
            logger.info("camera localization canceled")
            self.on_camera_localization_ended()

        self._task = worker.localize_pose.create_task(
            self._all_timestamps,
            self._marker_location_storage,
            self._markers_3d_model_storage,
            self._general_settings,
        )
        self._task.add_observer("on_yield", on_yield)
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_canceled_or_killed", on_canceled_or_killed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer("on_started", self.on_camera_localization_started)
        self._task_manager.add_task(self._task)
        logger.info("Start camera localization")
        self.status = "0% completed"

    def _insert_pose_bisector(self, data_pairs):
        for timestamp, pose in data_pairs:
            self._camera_localizer_storage.pose_bisector.insert(timestamp, pose)
        self.on_camera_localization_yield()

    def cancel_task(self):
        if self.is_running_task:
            self._task.kill(None)

    @property
    def is_running_task(self):
        return self._task is not None and self._task.running

    @property
    def localization_progress(self):
        return self._task.progress if self.is_running_task else 0.0

    def set_range_from_current_trim_marks(self):
        self._general_settings.camera_localizer_frame_index_range = (
            self._get_current_trim_mark_range()
        )

    def on_calculation_could_not_be_started(self):
        pass

    def on_camera_localization_started(self):
        pass

    def on_camera_localization_yield(self):
        pass

    def on_camera_localization_ended(self):
        pass
