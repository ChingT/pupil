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

        self._detection_task = None

    def start_detection(self):
        def on_detection_yields(marker_location):
            self._marker_location_storage[marker_location.frame_index] = marker_location

        def on_detection_completed(_):
            self._marker_location_storage.save_to_disk()

        self._detection_task = worker.detect_square_markers.SquareMarkerDetectionTask()
        self._detection_task.add_observer("on_exception", tasklib.raise_exception)
        self._detection_task.add_observer(
            "on_started", lambda: self.on_detection_started(self._detection_task)
        )
        self._detection_task.add_observer("on_yield", on_detection_yields)
        self._detection_task.add_observer("on_completed", on_detection_completed)
        self._task_manager.add_task(self._detection_task)
        return self._detection_task

    def on_detection_started(self, detection_task):
        """By observing this, other modules can add their own observers to the task"""
        pass

    def cancel_detection(self):
        if not self._detection_task:
            raise ValueError("No detection task running!")
        self._detection_task.cancel_gracefully()

    @property
    def is_running_detection(self):
        return self._detection_task is not None and self._detection_task.running

    @property
    def detection_progress(self):
        return self._detection_task.progress if self._detection_task else 0.0
