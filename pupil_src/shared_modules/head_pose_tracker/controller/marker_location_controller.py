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
        self, marker_location_storage, task_manager, get_current_trim_mark_range
    ):
        self._marker_location_storage = marker_location_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range

        self._marker_locations = marker_location_storage.item
        self._task = None

    def init_detection(self):
        if self._marker_locations.calculated:
            self.on_marker_detection_ended()
        else:
            self.start_detection()

    def start_detection(self):
        self._create_detection_task()

    def _create_detection_task(self):
        def on_yield(index_and_data):
            frame_index, detection_data = index_and_data
            self._marker_locations.result[frame_index] = detection_data
            self.on_marker_detection_yield()

        def on_completed(_):
            self._marker_location_storage.save_to_disk()
            logger.info("marker detection completed")
            self.on_marker_detection_ended()

        def on_canceled_or_killed():
            self._marker_location_storage.save_to_disk()
            logger.info("Cancel marker detection")
            self.on_marker_detection_ended()

        self._task = worker.detect_square_markers.create_task(self._marker_locations)
        self._task.add_observer("on_yield", on_yield)
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_canceled_or_killed", on_canceled_or_killed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer("on_started", self.on_marker_detection_started)
        self._task_manager.add_task(self._task)
        logger.info("Start marker detection")

    def cancel_detection(self):
        if self.is_running_detection:
            self._task.kill(None)

    @property
    def is_running_detection(self):
        return self._task is not None and self._task.running

    @property
    def detection_progress(self):
        return self._task.progress if self.is_running_detection else 0.0

    def set_range_from_current_trim_marks(self):
        self._marker_locations.frame_index_range = self._get_current_trim_mark_range()

    def on_marker_detection_started(self):
        pass

    def on_marker_detection_yield(self):
        pass

    def on_marker_detection_ended(self):
        pass
