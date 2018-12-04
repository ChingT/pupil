import logging

from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.marker_detector import MarkerDetector
from marker_tracker_3d.optimization.model_optimizer import ModelOptimizer

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, marker_tracker_3d, storage, camera_model, min_marker_perimeter):
        self.marker_tracker_3d = marker_tracker_3d
        self.storage = storage

        self.marker_detector = MarkerDetector(min_marker_perimeter)
        self.model_optimizer = ModelOptimizer(self.marker_tracker_3d, camera_model)
        self.camera_localizer = CameraLocalizer(camera_model)
        self.register_new_markers = True

        self.marker_tracker_3d.add_observer("recent_events", self.update)

        self.model_optimizer.add_observer(
            "got_marker_extrinsics", self._update_marker_extrinsics_in_storage
        )

    def update(self, events):
        frame = events.get("frame")

        self.storage.marker_detections = self.marker_detector.detect(frame)

        self.storage.camera_extrinsics = self.camera_localizer.get_camera_extrinsics(
            self.storage.marker_detections,
            self.storage.marker_extrinsics,
            self.storage.camera_extrinsics_previous,
        )

        if self.register_new_markers:
            self.model_optimizer.add_marker_detections(
                self.storage.marker_detections, self.storage.camera_extrinsics
            )

    def _update_marker_extrinsics_in_storage(self, extrinsics):
        if self.register_new_markers:
            self.storage.marker_extrinsics = extrinsics

    def on_restart(self):
        self.storage.reset()
        self.model_optimizer.model_optimizer_storage.reset()
        self.model_optimizer.restart()
        self.marker_tracker_3d.ui.update_menu()
        logger.info("Restart!")

    def on_export_data(self):
        self.storage.export_data()
        self.model_optimizer.model_optimizer_storage.export_data()
        logger.info("export data at {}".format(self.storage.save_path))
