import logging

from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.controller_storage import ControllerStorage
from marker_tracker_3d.marker_detector import MarkerDetector
from marker_tracker_3d.optimization.model_optimizer import ModelOptimizer

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, marker_tracker_3d, camera_model, min_marker_perimeter):
        self.marker_tracker_3d = marker_tracker_3d
        self.marker_tracker_3d.add_observer("recent_events", self._on_recent_events)

        user_dir = self.marker_tracker_3d.g_pool.user_dir

        self.storage = ControllerStorage(save_path=user_dir)
        self.marker_detector = MarkerDetector(min_marker_perimeter)
        self.model_optimizer = ModelOptimizer(
            self.marker_tracker_3d.plugin_task_manager, camera_model, save_path=user_dir
        )
        self.camera_localizer = CameraLocalizer(camera_model)

    def _on_recent_events(self, events):
        if "frame" in events:
            self._update(events["frame"])

    def get_init_dict(self):
        d = {"min_marker_perimeter": self.marker_detector.min_marker_perimeter}
        return d

    def _update(self, frame):
        self.storage.marker_id_to_detections = self.marker_detector.detect(frame)

        self.storage.current_camera_extrinsics = self.camera_localizer.get_current_camera_extrinsics(
            self.storage.marker_id_to_detections,
            self.model_optimizer.model_state.marker_extrinsics_opt_array,
        )

        self.model_optimizer.add_observations(
            self.storage.marker_id_to_detections, self.storage.current_camera_extrinsics
        )

    def reset(self):
        self.storage.reset()
        self.model_optimizer.reset()
        self.camera_localizer.reset()
        logger.info("Reset 3D Marker Tracker!")

    def export_marker_tracker_3d_model(self):
        self.model_optimizer.model_state.export_marker_tracker_3d_model()

    def export_camera_traces(self):
        self.storage.export_camera_traces()
