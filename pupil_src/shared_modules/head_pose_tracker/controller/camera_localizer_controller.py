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


class CameraLocalizerController(Observable):
    def __init__(
        self,
        camera_localizer_storage,
        optimization_storage,
        marker_location_storage,
        task_manager,
        get_current_trim_mark_range,
    ):
        self._camera_localizer_storage = camera_localizer_storage
        self._optimization_storage = optimization_storage
        self._marker_location_storage = marker_location_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range

    def set_mapping_range_from_current_trim_marks(self, camera_localizer):
        camera_localizer.mapping_index_range = self._get_current_trim_mark_range()

    def set_validation_range_from_current_trim_marks(self, camera_localizer):
        camera_localizer.validation_index_range = self._get_current_trim_mark_range()

    def calculate(self, camera_localizer):
        self._reset_camera_localizer_results(camera_localizer)
        optimization = self.get_valid_optimization_or_none(camera_localizer)
        if optimization is None:
            self._abort_calculation(
                camera_localizer,
                "The optimization was not found for the gaze mapper '{}', "
                "please select a different optimization!".format(camera_localizer.name),
            )
            return None
        if optimization.result is None:
            self._abort_calculation(
                camera_localizer,
                "You first need to calculate optimization '{}' before calculating the "
                "mapper '{}'".format(optimization.name, camera_localizer.name),
            )
            return None
        task = self._create_mapping_task(camera_localizer, optimization)
        self._task_manager.add_task(task)
        logger.info("Start gaze mapping for '{}'".format(camera_localizer.name))

    def _abort_calculation(self, camera_localizer, error_message):
        logger.error(error_message)
        camera_localizer.status = error_message
        self.on_calculation_could_not_be_started()
        # the gaze from this mapper got cleared, so don't show it anymore

    def on_calculation_could_not_be_started(self):
        pass

    def _reset_camera_localizer_results(self, camera_localizer):
        camera_localizer.gaze = []
        camera_localizer.gaze_ts = []
        camera_localizer.accuracy_result = ""
        camera_localizer.precision_result = ""

    def _create_mapping_task(self, camera_localizer, optimization):
        task = worker.map_gaze.create_task(
            camera_localizer, optimization, self._marker_location_storage
        )

        def on_yield_gaze(mapped_gaze_ts_and_data):
            camera_localizer.status = "Mapping {:.0f}% complete".format(
                task.progress * 100
            )
            for timestamp, gaze_datum in mapped_gaze_ts_and_data:
                camera_localizer.gaze.append(gaze_datum)
                camera_localizer.gaze_ts.append(timestamp)

        def on_completed_mapping(_):
            camera_localizer.status = "Successfully completed mapping"
            self._camera_localizer_storage.save_to_disk()
            self.on_camera_localization_calculated(camera_localizer)
            logger.info("Complete gaze mapping for '{}'".format(camera_localizer.name))

        task.add_observer("on_yield", on_yield_gaze)
        task.add_observer("on_completed", on_completed_mapping)
        task.add_observer("on_exception", tasklib.raise_exception)
        return task

    def on_camera_localization_calculated(self, camera_localizer):
        pass

    def get_valid_optimization_or_none(self, camera_localizer):
        return self._optimization_storage.get_or_none(
            camera_localizer.optimization_unique_id
        )
