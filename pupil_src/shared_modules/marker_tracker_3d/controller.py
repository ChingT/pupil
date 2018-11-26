import logging
import os

import numpy as np

from marker_tracker_3d import optimization
from marker_tracker_3d import utils
from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.marker_detector import MarkerDetector

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, storage, update_menu):
        self.storage = storage

        self.marker_detector = MarkerDetector(self.storage)
        self.optimization_controller = optimization.Controller(
            self.storage, update_menu
        )
        self.camera_localizer = CameraLocalizer(self.storage)

    def recent_events(self, frame):
        self.storage.marker_detections = self.marker_detector.detect(frame)

        self.storage.camera_extrinsics = self.camera_localizer.update(
            self.storage.marker_detections,
            self.storage.marker_extrinsics,
            self.storage.camera_extrinsics_previous,
        )

        if self.storage.register_new_markers:
            result = self.optimization_controller.update(
                self.storage.marker_detections, self.storage.camera_extrinsics
            )
            if result:
                self.storage.marker_extrinsics, self.storage.marker_points_3d = result

    def save_data(self):
        # For experiments
        if not os.path.exists(self.storage.save_path):
            os.makedirs(self.storage.save_path)

        dist = [
            np.linalg.norm(
                self.storage.camera_trace[i + 1] - self.storage.camera_trace[i]
            )
            if self.storage.camera_trace[i + 1] is not None
            and self.storage.camera_trace[i] is not None
            else np.nan
            for i in range(len(self.storage.camera_trace) - 1)
        ]

        dicts = {"dist": dist, "reprojection_errors": self.storage.reprojection_errors}
        utils.save_params_dicts(save_path=self.storage.save_path, dicts=dicts)

        logger.info("save_data at {}".format(self.storage.save_path))
        self.optimization_controller.save_data(self.storage.save_path)

    def restart(self):
        logger.info("Restart!")
        self.optimization_controller.restart()

    def cleanup(self):
        self.optimization_controller.cleanup()
