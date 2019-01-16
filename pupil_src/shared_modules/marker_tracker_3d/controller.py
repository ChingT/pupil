import logging


logger = logging.getLogger(__name__)


class Controller:
    def __init__(
        self,
        marker_detection_controller,
        model_optimization_controller,
        model_optimization_storage,
        camera_localization_controller,
        controller_storage,
        plugin,
    ):
        self._marker_detection_controller = marker_detection_controller
        self._model_optimization_controller = model_optimization_controller
        self._camera_localization_controller = camera_localization_controller

        self._model_optimization_storage = model_optimization_storage
        self._controller_storage = controller_storage

        plugin.add_observer("recent_events", self._on_recent_events)

    def _on_recent_events(self, events):
        if "frame" in events:
            self._update(events["frame"])

    def _update(self, frame):
        marker_id_to_detections = self._marker_detection_controller.detect(frame)

        current_camera_extrinsics = self._camera_localization_controller.get_current_camera_extrinsics(
            marker_id_to_detections,
            self._model_optimization_storage.marker_extrinsics_opt_array,
        )

        self._model_optimization_controller.add_observations(
            marker_id_to_detections, current_camera_extrinsics
        )

        self._update_storage(marker_id_to_detections, current_camera_extrinsics)

    def _update_storage(self, marker_id_to_detections, current_camera_extrinsics):
        self._controller_storage.marker_id_to_detections = marker_id_to_detections
        self._controller_storage.current_camera_extrinsics = current_camera_extrinsics

    def reset(self):
        self._controller_storage.reset()
        self._model_optimization_controller.reset()
        self._camera_localization_controller.reset()
        logger.info("Reset 3D Marker Tracker!")

    def export_marker_tracker_3d_model(self):
        self._model_optimization_storage.export_marker_tracker_3d_model()

    def export_camera_traces(self):
        self._controller_storage.export_camera_traces()
