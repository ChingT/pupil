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


class MarkerLocationController(Observable):
    def __init__(
        self,
        general_settings,
        marker_location_storage,
        task_manager,
        get_current_trim_mark_range,
        all_timestamps,
    ):
        self._general_settings = general_settings
        self._marker_location_storage = marker_location_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._all_timestamps = all_timestamps

        self._task = None

    def calculate(self):
        self._create_detection_task()

    def _create_detection_task(self):
        def on_yield(data_pairs):
            if data_pairs is None:
                # first yield (None), for instant update on menu and timeline
                self.on_marker_detection_yield()
            else:
                self._insert_markers_bisector(data_pairs)

        def on_completed(_):
            self._marker_location_storage.save_pldata_to_disk()
            logger.info("marker detection completed")
            self.on_marker_detection_ended()

        def on_canceled_or_killed():
            self._marker_location_storage.save_pldata_to_disk()
            logger.info("marker detection canceled")
            self.on_marker_detection_ended()

        self._task = worker.detect_square_markers.create_task(
            self._all_timestamps, self._general_settings, self._marker_location_storage
        )
        self._task.add_observer("on_yield", on_yield)
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_canceled_or_killed", on_canceled_or_killed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer("on_started", self.on_marker_detection_started)
        self._task_manager.add_task(self._task)
        logger.info("Start marker detection")

    def _insert_markers_bisector(self, data_pairs):
        for timestamp, markers, frame_index, num_markers in data_pairs:
            for marker in markers:
                self._marker_location_storage.markers_bisector.insert(timestamp, marker)
            self._marker_location_storage.frame_index_to_num_markers[
                frame_index
            ] = num_markers
        self.on_marker_detection_yield()

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
        self._general_settings.marker_location_frame_index_range = (
            self._get_current_trim_mark_range()
        )

    def on_marker_detection_started(self):
        pass

    def on_marker_detection_yield(self):
        pass

    def on_marker_detection_ended(self):
        pass
