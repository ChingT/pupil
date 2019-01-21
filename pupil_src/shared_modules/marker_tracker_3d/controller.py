import logging

from marker_tracker_3d import detect_markers
from marker_tracker_3d import localize_camera

logger = logging.getLogger(__name__)


class Controller:
    def __init__(
        self,
        visibility_graphs,
        initial_guess_controller,
        model_optimization_controller,
        model_optimization_storage,
        controller_storage,
        camera_model,
        plugin,
    ):
        self._visibility_graphs = visibility_graphs
        self._initial_guess_controller = initial_guess_controller
        self._model_optimization_controller = model_optimization_controller
        self._model_optimization_storage = model_optimization_storage
        self._controller_storage = controller_storage
        self._camera_model = camera_model

        plugin.add_observer("recent_events", self._on_recent_events)

        self._visibility_graphs.add_observer(
            "on_ready_for_initial_guess", self._initial_guess_controller.run
        )
        self._initial_guess_controller.add_observer(
            "on_got_data_for_opt", self._model_optimization_controller.run
        )
        self._initial_guess_controller.add_observer(
            "on_initial_guess_failed",
            self._visibility_graphs.switch_on_optimization_requested,
        )
        self._model_optimization_controller.add_observer(
            "on_optimization_done", self._visibility_graphs.process_optimization_results
        )

    def _on_recent_events(self, events):
        if "frame" in events:
            self._update(events["frame"])

    def _update(self, frame):
        # detect_markers, localize_camera, add_observations to visibility_graphs

        marker_id_to_detections = detect_markers.detect(
            frame, self._controller_storage.min_marker_perimeter
        )

        camera_extrinsics = localize_camera.localize(
            self._camera_model,
            marker_id_to_detections,
            self._model_optimization_storage.marker_id_to_extrinsics_opt,
            camera_extrinsics_prv=self._controller_storage.camera_extrinsics,
        )

        self._visibility_graphs.add_observations(
            marker_id_to_detections, camera_extrinsics
        )

        self._update_storage(marker_id_to_detections, camera_extrinsics)

    def _update_storage(self, marker_id_to_detections, camera_extrinsics):
        self._controller_storage.marker_id_to_detections = marker_id_to_detections
        self._controller_storage.camera_extrinsics = camera_extrinsics

    def reset(self):
        self._visibility_graphs.reset()
        self._model_optimization_controller.reset()
        self._controller_storage.reset()
        self._model_optimization_storage.reset()

        logger.info("Reset 3D Marker Tracker!")

    def export_marker_tracker_3d_model(self):
        self._model_optimization_storage.export_marker_tracker_3d_model()

    def export_camera_traces(self):
        self._controller_storage.export_camera_traces()
