import logging

from marker_tracker_3d import detect_markers
from marker_tracker_3d import localize_camera
from marker_tracker_3d.model_initialization_controller import (
    ModelInitializationController,
)
from marker_tracker_3d.optimization.model_optimization_controller import (
    ModelOptimizationController,
)
from marker_tracker_3d.optimization.prepare_for_model_update import (
    PrepareForModelUpdate,
)
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs
from marker_tracker_3d.update_model_storage import UpdateModelStorage
from observable import Observable

logger = logging.getLogger(__name__)


class Controller(Observable):
    def __init__(
        self, controller_storage, model_storage, camera_intrinsics, task_manager, plugin
    ):
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._camera_intrinsics = camera_intrinsics
        self._plugin = plugin

        self._visibility_graphs = VisibilityGraphs(model_storage)
        self._prepare_for_model_update = PrepareForModelUpdate(
            model_storage, predetermined_origin_marker_id=None
        )
        self._model_initialization_controller = ModelInitializationController(
            camera_intrinsics, task_manager
        )
        self._model_optimization_controller = ModelOptimizationController(
            model_storage, camera_intrinsics, task_manager
        )
        self._update_model_storage = UpdateModelStorage(model_storage)

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
        marker_id_to_detections = detect_markers.detect(
            frame, self._controller_storage.min_marker_perimeter
        )
        camera_extrinsics = localize_camera.localize(
            self._camera_intrinsics,
            marker_id_to_detections,
            self._model_storage.marker_id_to_extrinsics_opt,
            camera_extrinsics_prv=self._controller_storage.camera_extrinsics,
        )
        self._controller_storage.update(marker_id_to_detections, camera_extrinsics)

    def _on_update_model_started(self, args=None):
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

    def export_marker_tracker_3d_model(self):
        self._model_storage.export_marker_tracker_3d_model()

    def export_camera_traces(self):
        self._controller_storage.export_camera_traces()
