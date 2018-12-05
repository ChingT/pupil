import logging

from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.controller_storage import ControllerStorage
from marker_tracker_3d.marker_detector import MarkerDetector
from marker_tracker_3d.optimization.model_optimizer import ModelOptimizer

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, marker_tracker_3d, camera_model, min_marker_perimeter):
        self.marker_tracker_3d = marker_tracker_3d
        self.storage = ControllerStorage()

        self.marker_detector = MarkerDetector(min_marker_perimeter)
        self.model_optimizer = ModelOptimizer(self.marker_tracker_3d, camera_model)
        self.camera_localizer = CameraLocalizer(camera_model)
        self.register_new_markers = True

        self.marker_tracker_3d.add_observer("recent_events", self.update)

    def update(self, events):
        frame = events.get("frame")

        self.storage.marker_detections = self.marker_detector.detect(frame)

        self.storage.camera_extrinsics = self.camera_localizer.get_camera_extrinsics(
            self.storage.marker_detections,
            self.model_optimizer.storage.marker_extrinsics_opt,
        )

        if self.register_new_markers:
            self.model_optimizer.add_marker_detections(
                self.storage.marker_detections, self.storage.camera_extrinsics
            )

    def on_restart(self):
        self.storage.reset()
        self.model_optimizer.restart()
        self.camera_localizer.reset()
        logger.info("Restart!")

    def on_export_data(self):
        self.model_optimizer.storage.export_data()
        logger.info("export data at {}".format(self.model_optimizer.storage.save_path))
