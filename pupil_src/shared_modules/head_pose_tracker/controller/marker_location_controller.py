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
    def __init__(self, marker_location_storage, task_manager):
        self._marker_location_storage = marker_location_storage
        self._task_manager = task_manager
        self._task = None

    def start_detection(self):
        self._create_detection_task()

    def _create_detection_task(self):
        def on_yield_location(marker_location):
            self._marker_location_storage[marker_location.frame_index] = marker_location
            self.on_detection_yield()

        def on_completed_location(_):
            self._marker_location_storage.save_to_disk()
            logger.info("Complete marker detection")
            self.on_detection_ended()

        def on_canceled_or_killed():
            self._marker_location_storage.save_to_disk()
            logger.info("Cancel marker detection")
            self.on_detection_ended()

        self._task = worker.detect_square_markers.create_task()
        self._task.add_observer("on_yield", on_yield_location)
        self._task.add_observer("on_completed", on_completed_location)
        self._task.add_observer("on_canceled_or_killed", on_canceled_or_killed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task_manager.add_task(self._task)
        logger.info("Start marker detection")
        self.on_detection_started()

    def on_detection_started(self):
        pass

    def on_detection_yield(self):
        pass

    def on_detection_ended(self):
        pass

    def cancel_detection(self):
        if not self._task:
            raise ValueError("No detection task running!")
        self._task.cancel_gracefully()

    @property
    def is_running_detection(self):
        return self._task is not None and self._task.running

    @property
    def detection_progress(self):
        return self._task.progress if self._task else 0.0
