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
        marker_location_storage,
        markers_3d_model_storage,
        camera_localizer_storage,
        task_manager,
        get_current_trim_mark_range,
    ):
        self._markers_3d_model_storage = markers_3d_model_storage
        self._camera_localizer_storage = camera_localizer_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range

        self._camera_localizer = camera_localizer_storage.item
        self._marker_locations = marker_location_storage.item

        self._task = None
        self.pose = []
        self.pose_ts = []

        markers_3d_model_controller.add_observer(
            "on_building_markers_3d_model_had_completed_before",
            self._on_building_markers_3d_model_had_completed_before,
        )
        markers_3d_model_controller.add_observer(
            "on_building_markers_3d_model_started",
            self._on_building_markers_3d_model_started,
        )
        markers_3d_model_controller.add_observer(
            "on_building_markers_3d_model_completed",
            self._on_building_markers_3d_model_completed,
        )

    def _on_building_markers_3d_model_had_completed_before(self):
        if not self._camera_localizer.calculate_complete:
            self.calculate()

    def _on_building_markers_3d_model_started(self):
        self._reset()

    def _on_building_markers_3d_model_completed(self):
        self.calculate()

    def calculate(self):
        markers_3d_model = self._get_valid_markers_3d_model_or_none()
        if markers_3d_model is None:
            return

        self._reset()
        self._create_localization_task(markers_3d_model)

    def _get_valid_markers_3d_model_or_none(self):
        markers_3d_model = self._markers_3d_model_storage.item
        if not markers_3d_model.result:
            self._abort_calculation(
                "You first need to calculate markers_3d_model '{}' before calculating "
                "the camera localizer".format(markers_3d_model.name)
            )
            return None
        return markers_3d_model

    def _abort_calculation(self, error_message):
        logger.error(error_message)
        self._camera_localizer.status = error_message
        self.on_calculation_could_not_be_started()
        # the pose from this localizer got cleared, so don't show it anymore
        self.save_pose_bisector()

    def _reset(self):
        if self._task is not None and self._task.running:
            self._task.kill(None)

        self.pose = []
        self.pose_ts = []

        self.save_pose_bisector()
        self._camera_localizer.status = "Not calculated yet"

    def _create_localization_task(self, markers_3d_model):
        def on_yield_pose(localized_pose_ts_and_data):
            self._camera_localizer.status = (
                "Calculating localization {:.0f}% "
                "complete".format(self._task.progress * 100)
            )
            self._update_result(localized_pose_ts_and_data)

        def on_completed_localization(_):
            self._camera_localizer.status = "Calculating localization successfully"
            self.save_pose_bisector()
            self._camera_localizer_storage.save_to_disk()
            logger.info("Complete camera localization")
            self.on_camera_localization_completed()

        self._task = worker.localize_pose.create_task(
            self._camera_localizer, markers_3d_model, self._marker_locations.detections
        )
        self._task.add_observer("on_yield", on_yield_pose)
        self._task.add_observer("on_completed", on_completed_localization)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task_manager.add_task(self._task)
        logger.info("Start camera localization")

    def _update_result(self, localized_pose_ts_and_data):
        for timestamp, pose_datum in localized_pose_ts_and_data:
            self.pose.append(pose_datum)
            self.pose_ts.append(timestamp)

    def save_pose_bisector(self):
        self._camera_localizer.pose_bisector = pm.Bisector(self.pose, self.pose_ts)

    def set_range_from_current_trim_marks(self):
        self._camera_localizer.localization_index_range = (
            self._get_current_trim_mark_range()
        )

    def on_calculation_could_not_be_started(self):
        pass

    def on_camera_localization_completed(self):
        pass
