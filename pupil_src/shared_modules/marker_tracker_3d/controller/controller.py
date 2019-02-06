import logging

from marker_tracker_3d import worker, controller
from observable import Observable

logger = logging.getLogger(__name__)


class Controller(Observable):
    def __init__(
        self,
        controller_storage,
        model_storage,
        camera_intrinsics,
        task_manager,
        plugin,
        save_path,
    ):
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._camera_intrinsics = camera_intrinsics
        self._plugin = plugin
        self._save_path = save_path

        self._visibility_graphs = worker.VisibilityGraphs(model_storage)
        self._prepare_for_model_update = worker.PrepareForModelUpdate(
            model_storage, predetermined_origin_marker_id=None
        )
        self._model_initialization_controller = controller.ModelInitializationController(
            camera_intrinsics, task_manager
        )
        self._model_optimization_controller = controller.ModelOptimizationController(
            model_storage, camera_intrinsics, task_manager
        )
        self._update_model_storage = worker.UpdateModelStorage(
            model_storage, camera_intrinsics
        )

        self._plugin.add_observer("recent_events", self._on_recent_events)
        self._setup_model_update_pipeline()

    def _setup_model_update_pipeline(self):
        self._controller_storage.add_observer(
            "update", self._visibility_graphs.add_observations
        )
        self._visibility_graphs.add_observer(
            "on_novel_markers_added", self._prepare_for_model_update.run
        )
        self._prepare_for_model_update.add_observer(
            "on_prepare_for_model_init_done", self._on_update_model_started
        )
        self.add_observer(
            "_on_update_model_started", self._model_initialization_controller.run
        )
        self._model_initialization_controller.add_observer(
            "on_model_init_done", self._model_optimization_controller.run
        )
        # TODO: debug only; to be removed
        self._model_initialization_controller.add_observer(
            "on_model_init_done", self._update_model_storage.run_init
        )
        self._model_optimization_controller.add_observer(
            "on_model_opt_done", self._update_model_storage.run
        )
        self._update_model_storage.add_observer(
            "on_update_model_storage_done", self._on_update_model_finished
        )
        self.add_observer(
            "_on_update_model_finished", self._prepare_for_model_update.run
        )

    def _update(self, frame):
        marker_id_to_detections = worker.detect_markers.detect(
            frame, self._controller_storage.min_marker_perimeter
        )
        camera_extrinsics = worker.localize_camera.localize(
            self._camera_intrinsics,
            marker_id_to_detections,
            self._model_storage.marker_id_to_extrinsics_opt,
            self._controller_storage.camera_extrinsics,
            self._model_storage.origin_marker_id,
        )
        self._controller_storage.update(marker_id_to_detections, camera_extrinsics)

    def _on_update_model_started(self, _):
        self._model_storage.model_being_updated = True

    def _on_update_model_finished(self):
        self._model_storage.model_being_updated = False

    def _on_recent_events(self, events):
        if "frame" in events:
            self._update(events["frame"])

    def reset(self):
        self._controller_storage.reset()
        self._model_storage.reset()
        self._visibility_graphs.reset()
        self._model_initialization_controller.reset()
        self._model_optimization_controller.reset()
        logger.info("Reset 3D Marker Tracker!")

    def load_marker_tracker_3d_model(self):
        self._model_storage.load_marker_tracker_3d_model_from_file()

    def export_marker_tracker_3d_model(self):
        self._model_storage.export_marker_tracker_3d_model()

    # TODO: debug only; to be removed
    def export_visibility_graph(self):
        self._model_storage.export_visibility_graph()

    # TODO: maybe should be moved to other place
    def export_camera_intrinsics(self):
        self._camera_intrinsics.save(self._save_path)

    def export_camera_traces(self):
        self._controller_storage.export_camera_traces()
