import logging

from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.controller_storage import ControllerStorage
from marker_tracker_3d.marker_detector import MarkerDetector
from marker_tracker_3d.optimization.model_optimizer import ModelOptimizer

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, marker_tracker_3d, camera_model, min_marker_perimeter):
        self.marker_tracker_3d = marker_tracker_3d
        self.marker_tracker_3d.add_observer("recent_events", self.update)

        root = self.marker_tracker_3d.g_pool.user_dir

        self.storage = ControllerStorage(save_path=root)
        self.marker_detector = MarkerDetector(min_marker_perimeter)
        self.model_optimizer = ModelOptimizer(
            self.marker_tracker_3d.plugin_task_manager, camera_model, save_path=root
        )
        self.camera_localizer = CameraLocalizer(camera_model)

    def update(self, events):
        frame = events.get("frame")
        if not frame:
            return

        self.storage.current_marker_detections = self.marker_detector.detect(frame)

        self.storage.current_camera_extrinsics = self.camera_localizer.get_camera_extrinsics(
            self.storage.current_marker_detections,
            self.model_optimizer.model_state.marker_extrinsics_opt,
        )

        self.model_optimizer.add_observations(
            self.storage.current_marker_detections,
            self.storage.current_camera_extrinsics,
        )

    def on_reset(self):
        self.storage.reset()
        self.model_optimizer.reset()
        self.camera_localizer.reset()
        logger.info("reset!")

    def on_export_marker_tracker_3d_model(self):
        self.model_optimizer.model_state.export_marker_tracker_3d_model()

    def on_export_camera_traces(self):
        self.storage.export_camera_traces()
