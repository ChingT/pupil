import logging
import os

import numpy as np

from marker_tracker_3d import optimization
from marker_tracker_3d import utils
from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.marker_detector import MarkerDetector

logger = logging.getLogger(__name__)


class Controller:
    def __init__(
        self, storage, camera_model, marker_model, update_menu, min_marker_perimeter
    ):
        self.storage = storage

        self.marker_detector = MarkerDetector(min_marker_perimeter)
        self.optimization_controller = optimization.Controller(
            camera_model, marker_model, update_menu
        )
        self.camera_localizer = CameraLocalizer(camera_model, marker_model)
        self.register_new_markers = True

    def recent_events(self, frame):
        self.storage.markers = self.marker_detector.detect(frame)

        self.storage.camera_extrinsics = self.camera_localizer.get_camera_extrinsics(
            self.storage.markers,
            self.storage.marker_extrinsics,
            self.storage.camera_extrinsics_previous,
        )

        if self.register_new_markers:
            self.storage.marker_extrinsics, self.storage.marker_points_3d = self.optimization_controller.update(
                self.storage.markers, self.storage.camera_extrinsics
            )

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
