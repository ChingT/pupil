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
        marker_location_storage,
        task_manager,
        get_current_trim_mark_range,
        all_timestamps,
    ):
        self._marker_location_storage = marker_location_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._all_timestamps = all_timestamps

        self._marker_locations = marker_location_storage.item
        self._task = None
        self._set_to_default_values()

    def _set_to_default_values(self):
        self._timestamps_queue = []
        self._markers_queue = []
        self._calculated_count = 0
        self._last_count = 0

    def init_detection(self):
        if self._marker_locations.calculated:
            self.on_marker_detection_ended()
        else:
            self.calculate()

    def calculate(self):
        self._set_to_default_values()
        self._create_detection_task()

    def _create_detection_task(self):
        def on_yield(ts_idx_data):
            if ts_idx_data:
                self._add_data_to_queue(ts_idx_data)
                self._insert_markers_bisector(force=False)
            else:
                # first yield (None), for instant update on menu and timeline
                self.on_marker_detection_yield()

        def on_completed(_):
            self._insert_markers_bisector(force=True)
            self._marker_location_storage.save_to_disk()
            logger.info("marker detection completed")
            self.on_marker_detection_ended()

        def on_canceled_or_killed():
            self._insert_markers_bisector(force=True)
            self._marker_location_storage.save_to_disk()
            logger.info("marker detection canceled")
            self.on_marker_detection_ended()

        self._task = worker.detect_square_markers.create_task(
            self._all_timestamps, self._marker_locations
        )
        self._task.add_observer("on_yield", on_yield)
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_canceled_or_killed", on_canceled_or_killed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer("on_started", self.on_marker_detection_started)
        self._task_manager.add_task(self._task)
        logger.info("Start marker detection")

    def _add_data_to_queue(self, ts_idx_data):
        self._calculated_count += 1
        timestamp, frame_index, markers = ts_idx_data
        self._marker_locations.calculated_frame_indices.append(frame_index)
        for marker in markers:
            self._timestamps_queue.append(timestamp)
            self._markers_queue.append(marker)

    def _insert_markers_bisector(self, force):
        if force or self._calculated_count - self._last_count > 150:
            self._last_count = self._calculated_count
            for timestamp, marker in zip(self._timestamps_queue, self._markers_queue):
                self._marker_locations.markers_bisector.insert(timestamp, marker)
            self._timestamps_queue = []
            self._markers_queue = []
            self.on_marker_detection_yield()

    def cancel_task(self):
        if self.is_running_task:
            self._task.kill(None)

    @property
    def is_running_task(self):
        return self._task is not None and self._task.running

    @property
    def detection_progress(self):
        return self._task.progress if self.is_running_task else 0.0

    def set_range_from_current_trim_marks(self):
        self._marker_locations.frame_index_range = self._get_current_trim_mark_range()

    def on_marker_detection_started(self):
        pass

    def on_marker_detection_yield(self):
        pass

    def on_marker_detection_ended(self):
        pass
