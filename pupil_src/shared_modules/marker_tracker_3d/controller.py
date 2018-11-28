import logging

from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.marker_detector import MarkerDetector
from marker_tracker_3d.optimization.model_optimizer import ModelOptimizer

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, storage, camera_model, update_menu, min_marker_perimeter):
        self.storage = storage

        self.marker_detector = MarkerDetector(min_marker_perimeter)
        self.model_optimizer = ModelOptimizer(camera_model, update_menu)
        self.camera_localizer = CameraLocalizer(camera_model)
        self.register_new_markers = True

    def update(self, frame):
        self.storage.marker_detections = self.marker_detector.detect(frame)

        self.storage.camera_extrinsics = self.camera_localizer.get_camera_extrinsics(
            self.storage.marker_detections,
            self.storage.marker_extrinsics,
            self.storage.camera_extrinsics_previous,
        )

        if self.register_new_markers:
            result = self.model_optimizer.update(
                self.storage.marker_detections, self.storage.camera_extrinsics
            )
            if result:
                self.storage.marker_extrinsics, self.storage.marker_points_3d = result

    def export_data(self):
        logger.info("save_data at {}".format(self.storage.save_path))
        self.storage.export_data()
        self.model_optimizer.export_data(self.storage.save_path)

    def restart(self):
        logger.info("Restart!")
        self.model_optimizer.restart()

    def cleanup(self):
        self.model_optimizer.cleanup()
