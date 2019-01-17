import logging

from marker_tracker_3d import detect_markers
from marker_tracker_3d import localize_camera

logger = logging.getLogger(__name__)


class Controller:
    def __init__(
        self,
        model_optimization_controller,
        model_optimization_storage,
        controller_storage,
        camera_model,
        plugin,
    ):
        self._model_optimization_controller = model_optimization_controller
        self._model_optimization_storage = model_optimization_storage
        self._controller_storage = controller_storage
        self._camera_model = camera_model

        plugin.add_observer("recent_events", self._on_recent_events)

    def _on_recent_events(self, events):
        if "frame" in events:
            self._update(events["frame"])

    def _update(self, frame):
        marker_id_to_detections = detect_markers.detect(
            frame, self._controller_storage.min_marker_perimeter
        )

        data = (
            marker_id_to_detections,
            self._model_optimization_storage.marker_extrinsics_opt_dict,
        )
        camera_extrinsics = localize_camera.get(
            self._camera_model,
            data,
            camera_extrinsics_prv=self._controller_storage.camera_extrinsics,
        )

        self._model_optimization_controller.add_observations(
            marker_id_to_detections, camera_extrinsics
        )

        self._update_storage(marker_id_to_detections, camera_extrinsics)

    def _update_storage(self, marker_id_to_detections, camera_extrinsics):
        self._controller_storage.marker_id_to_detections = marker_id_to_detections
        self._controller_storage.camera_extrinsics = camera_extrinsics

    def reset(self):
        self._controller_storage.reset()
        self._model_optimization_controller.reset()
        logger.info("Reset 3D Marker Tracker!")

    def export_marker_tracker_3d_model(self):
        self._model_optimization_storage.export_marker_tracker_3d_model()

    def export_camera_traces(self):
        self._controller_storage.export_camera_traces()
