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

import head_pose_tracker.model.marker_location_storage
import zmq_tools
from tasklib.interface import TaskInterface

logger = logging.getLogger(__name__)


class SquareMarkerDetectionTask(TaskInterface):
    """
    The actual marker detection is in launchabeles.marker_detector because OpenCV
    needs a new and clean process and does not work with forked processes.

    This task requests the start of the launchable via a notification and retrieves
    results from the background. It does _not_ run in background itself, but does its
    work in the update() method that is executed in the main process.
    """

    # plugin injected variables
    zmq_ctx = None
    capture_source_path = None
    notify_all = None

    def __init__(self):
        super().__init__()
        self._process_pipe = None
        self._progress = 0.0

    @property
    def progress(self):
        return self._progress

    def start(self):
        super().start()
        self._process_pipe = zmq_tools.Msg_Pair_Server(self.zmq_ctx)
        self._request_start_of_detection(self._process_pipe.url)

    def _request_start_of_detection(self, pair_url):
        source_path = self.capture_source_path
        self.notify_all(
            {
                "subject": "square_detector_process.should_start",
                "source_path": source_path,
                "pair_url": pair_url,
            }
        )

    def cancel_gracefully(self):
        super().cancel_gracefully()
        self._terminate_background_detection()
        self.on_canceled_or_killed()

    def kill(self, grace_period):
        super().kill(grace_period)
        # we cannot kill the detection, just ask it to terminate
        self._terminate_background_detection()
        self.on_canceled_or_killed()

    def _terminate_background_detection(self):
        self._process_pipe.send({"topic": "terminate"})
        self._process_pipe.socket.close()
        self._process_pipe = None

    def update(self):
        super().update()
        try:
            self._receive_detections()
        except Exception as e:
            self.on_exception(e)

    def _receive_detections(self):
        while self._process_pipe.new_data:
            topic, msg = self._process_pipe.recv()
            if topic == "progress":
                progress_detection_pairs = msg.get("data", [])
                progress, detections = zip(*progress_detection_pairs)
                self._progress = progress[-1] / 100.0
                detections_without_None_items = (d for d in detections if d)
                for detection in detections_without_None_items:
                    self._yield_detection(detection)
            elif topic == "finished":
                self.on_completed(None)  # return_value_or_none
                return
            elif topic == "exception":
                logger.warning(
                    "Markers3DModel marker detection raised exception:\n{}".format(
                        msg["reason"]
                    )
                )
                logger.info("Marker detection was interrupted")
                logger.debug("Reason: {}".format(msg.get("reason", "n/a")))
                self.on_canceled_or_killed()
                return

    def _yield_detection(self, detection):
        marker_detection = detection["marker_detection"]
        frame_index = detection["index"]
        timestamp = detection["timestamp"]
        marker_location = head_pose_tracker.model.marker_location_storage.MarkerLocation(
            marker_detection, frame_index, timestamp
        )
        self.on_yield(marker_location)
