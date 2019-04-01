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
        self._marker_location_storage = marker_location_storage
        self._markers_3d_model_storage = markers_3d_model_storage
        self._camera_localizer_storage = camera_localizer_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range

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
        camera_localizer = self._camera_localizer_storage.item
        if not camera_localizer.calculate_complete:
            self.calculate(camera_localizer)

    def _on_building_markers_3d_model_started(self):
        camera_localizer = self._camera_localizer_storage.item
        self._reset(camera_localizer)

    def _on_building_markers_3d_model_completed(self):
        camera_localizer = self._camera_localizer_storage.item
        self.calculate(camera_localizer)

    def calculate(self, camera_localizer):
        markers_3d_model = self._get_valid_markers_3d_model_or_none(camera_localizer)
        if markers_3d_model is None:
            return

        self._reset(camera_localizer)
        self._create_localization_task(camera_localizer, markers_3d_model)

    def _get_valid_markers_3d_model_or_none(self, camera_localizer):
        markers_3d_model = self._markers_3d_model_storage.item
        if not markers_3d_model.result:
            self._abort_calculation(
                camera_localizer,
                "You first need to calculate markers_3d_model '{}' before calculating "
                "the camera localizer".format(markers_3d_model.name),
            )
            return None
        return markers_3d_model

    def _abort_calculation(self, camera_localizer, error_message):
        logger.error(error_message)
        camera_localizer.status = error_message
        self.on_calculation_could_not_be_started()
        # the pose from this localizer got cleared, so don't show it anymore
        self.save_pose_bisector(camera_localizer)

    def _reset(self, camera_localizer):
        if self._task is not None and self._task.running:
            self._task.kill(None)

        self.pose = []
        self.pose_ts = []

        self.save_pose_bisector(camera_localizer)
        camera_localizer.status = "Not calculated yet"

    def _create_localization_task(self, camera_localizer, markers_3d_model):
        def on_yield_pose(localized_pose_ts_and_data):
            camera_localizer.status = (
                "Calculating localization {:.0f}% "
                "complete".format(self._task.progress * 100)
            )
            self._update_result(localized_pose_ts_and_data)

        def on_completed_localization(_):
            camera_localizer.status = "Calculating localization successfully"
            self.save_pose_bisector(camera_localizer)
            self._camera_localizer_storage.save_to_disk()
            logger.info("Complete camera localization")
            self.on_camera_localization_completed()

        self._task = worker.localize_pose.create_task(
            camera_localizer, markers_3d_model, self._marker_location_storage
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

    def save_pose_bisector(self, camera_localizer):
        camera_localizer.pose_bisector = pm.Bisector(self.pose, self.pose_ts)

    def set_range_from_current_trim_marks(self, camera_localizer):
        camera_localizer.localization_index_range = self._get_current_trim_mark_range()

    def on_calculation_could_not_be_started(self):
        pass

    def on_camera_localization_completed(self):
        pass
