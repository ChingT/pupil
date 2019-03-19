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
        camera_localizer_storage,
        markers_3d_model_storage,
        marker_location_storage,
        task_manager,
        get_current_trim_mark_range,
    ):
        self._markers_3d_model_controller = markers_3d_model_controller
        self._camera_localizer_storage = camera_localizer_storage
        self._markers_3d_model_storage = markers_3d_model_storage
        self._marker_location_storage = marker_location_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range

        # make localizations loaded from disk known to Player
        self.save_all_enabled_localizers()

        self._markers_3d_model_controller.add_observer(
            "on_markers_3d_model_computed", self.calculate
        )

    def set_localization_range_from_current_trim_marks(self, camera_localizer):
        camera_localizer.localization_index_range = self._get_current_trim_mark_range()

    def calculate(self, markers_3d_model=None):
        camera_localizer = self._camera_localizer_storage.get_or_none()
        if camera_localizer is None:
            return

        self._reset_camera_localizer_results(camera_localizer)

        if markers_3d_model is None:
            markers_3d_model = self.get_valid_markers_3d_model_or_none()

        if markers_3d_model is None:
            self._abort_calculation(
                camera_localizer,
                "The markers_3d_model was not found for the pose localizer "
                "'{}'".format(camera_localizer.name),
            )
            return None
        if markers_3d_model.result is None:
            self._abort_calculation(
                camera_localizer,
                "You first need to calculate markers_3d_model '{}' before calculating the "
                "localizer '{}'".format(markers_3d_model.name, camera_localizer.name),
            )
            return None
        task = self._create_localization_task(camera_localizer, markers_3d_model)
        self._task_manager.add_task(task)
        logger.info("Start pose localization for '{}'".format(camera_localizer.name))

    def _abort_calculation(self, camera_localizer, error_message):
        logger.error(error_message)
        camera_localizer.status = error_message
        self.on_calculation_could_not_be_started()
        # the pose from this localizer got cleared, so don't show it anymore
        self.save_all_enabled_localizers()

    def on_calculation_could_not_be_started(self):
        pass

    def _reset_camera_localizer_results(self, camera_localizer):
        camera_localizer.pose = []
        camera_localizer.pose_ts = []

    def _create_localization_task(self, camera_localizer, markers_3d_model):
        task = worker.localize_pose.create_task(
            camera_localizer, markers_3d_model, self._marker_location_storage
        )

        def on_yield_pose(localized_pose_ts_and_data):
            camera_localizer.status = (
                "Calculating localization {:.0f}% "
                "complete".format(task.progress * 100)
            )
            for timestamp, pose_datum in localized_pose_ts_and_data:
                camera_localizer.pose.append(pose_datum)
                camera_localizer.pose_ts.append(timestamp)

        def on_completed_localization(_):
            camera_localizer.status = "Calculating localization successfully"
            self.save_all_enabled_localizers()
            self._camera_localizer_storage.save_to_disk()
            self.on_camera_localization_calculated(camera_localizer)
            logger.info(
                "Complete camera localization for '{}'".format(camera_localizer.name)
            )

        task.add_observer("on_yield", on_yield_pose)
        task.add_observer("on_completed", on_completed_localization)
        task.add_observer("on_exception", tasklib.raise_exception)
        return task

    def save_all_enabled_localizers(self):
        """
        Save pose data to e.g. render it in Player or to trigger other plugins
        that operate on pose data. The save logic is implemented in the plugin.
        """
        camera_localizer = self._camera_localizer_storage.get_or_none()
        if camera_localizer is None:
            return

        pose_bisector = self._create_pose_bisector_from_localizer(camera_localizer)
        self._camera_localizer_storage.save_pose_bisector(
            camera_localizer, pose_bisector
        )

    def _create_pose_bisector_from_localizer(self, camera_localizer):
        pose_data = list(camera_localizer.pose)
        pose_ts = list(camera_localizer.pose_ts)
        return pm.Bisector(pose_data, pose_ts)

    def on_camera_localization_calculated(self, camera_localizer):
        pass

    def get_valid_markers_3d_model_or_none(self):
        return self._markers_3d_model_storage.get_or_none()
