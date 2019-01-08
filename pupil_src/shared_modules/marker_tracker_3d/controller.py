import logging
import os

from marker_tracker_3d import utils
from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.controller_storage import ControllerStorage
from marker_tracker_3d.marker_detector import MarkerDetector
from marker_tracker_3d.optimization.model_optimizer import ModelOptimizer

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, marker_tracker_3d, camera_model, min_marker_perimeter):
        self.marker_tracker_3d = marker_tracker_3d
        self.marker_tracker_3d.add_observer("recent_events", self.update)

        root = os.path.join(
            self.marker_tracker_3d.g_pool.user_dir, "plugins", "marker_tracker_3d"
        )
        self.save_path = utils.get_save_path(root)

        self.storage = ControllerStorage(save_path=root)
        self.marker_detector = MarkerDetector(min_marker_perimeter)
        self.model_optimizer = ModelOptimizer(
            self.marker_tracker_3d.plugin_task_manager,
            camera_model,
            save_path=self.save_path,
        )
        self.camera_localizer = CameraLocalizer(camera_model)

    def update(self, events):
        frame = events.get("frame")
        if not frame:
            return

        self.storage.marker_detections = self.marker_detector.detect(frame)

        self.storage.camera_extrinsics = self.camera_localizer.get_camera_extrinsics(
            self.storage.marker_detections,
            self.model_optimizer.storage.marker_extrinsics_opt,
        )

        self.model_optimizer.add_observations(
            self.storage.marker_detections, self.storage.camera_extrinsics
        )

    def on_restart(self):
        self.storage.reset()
        self.model_optimizer.restart()
        self.camera_localizer.reset()
        logger.info("Restart!")

    def on_export_data(self):
        self.model_optimizer.storage.export_data()
        self.storage.export_data()
