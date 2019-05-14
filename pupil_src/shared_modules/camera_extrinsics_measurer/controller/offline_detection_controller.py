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
from camera_extrinsics_measurer import worker
from observable import Observable

logger = logging.getLogger(__name__)


class OfflineDetectionController(Observable):
    def __init__(
        self,
        general_settings,
        detection_storage,
        task_manager,
        current_trim_mark_ts_range,
        all_timestamps_dict,
        source_path_dict,
    ):
        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._task_manager = task_manager
        self._current_trim_mark_ts_range = current_trim_mark_ts_range
        self._all_timestamps_dict = all_timestamps_dict
        self._source_path_dict = source_path_dict
        self._task = None

    def calculate(self, camera_name):
        self._create_detection_task(camera_name)

    def _create_detection_task(self, camera_name):
        def on_yield(data_pairs):
            if data_pairs is None:
                # first yield (None), for instant update on menu and timeline
                self.on_detection_yield()
            else:
                self._insert_markers_bisector(camera_name, data_pairs)

        def on_completed(_):
            self._detection_storage.save_pldata_to_disk()
            logger.info("[{}] marker detection completed".format(camera_name))
            self.on_detection_ended(camera_name)

        def on_canceled_or_killed():
            self._detection_storage.save_pldata_to_disk()
            logger.info("[{}] marker detection canceled".format(camera_name))
            self.on_detection_ended(camera_name)

        self._task = self._create_task(camera_name)
        self._task.add_observer("on_yield", on_yield)
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_canceled_or_killed", on_canceled_or_killed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer("on_started", self.on_detection_started)
        logger.info("[{}] Start marker detection".format(camera_name))

    def _create_task(self, camera_name):
        args = (
            self._source_path_dict[camera_name],
            self._general_settings.detection_frame_ts_range,
            self._all_timestamps_dict[camera_name],
            self._detection_storage.frame_index_to_num_markers[camera_name],
        )
        return self._task_manager.create_background_task(
            name="marker detection",
            routine_or_generator_function=worker.offline_detection,
            pass_shared_memory=True,
            args=args,
        )

    def _insert_markers_bisector(self, camera_name, data_pairs):
        for timestamp, markers, frame_index, num_markers in data_pairs:
            for marker in markers:
                self._detection_storage.markers_bisector[camera_name].insert(
                    timestamp, marker
                )
            self._detection_storage.frame_index_to_num_markers[camera_name][
                frame_index
            ] = num_markers
        self.on_detection_yield()

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
        self._general_settings.detection_frame_ts_range = (
            self._current_trim_mark_ts_range()
        )

    def on_detection_started(self):
        pass

    def on_detection_yield(self):
        pass

    def on_detection_ended(self, camera_name):
        pass
