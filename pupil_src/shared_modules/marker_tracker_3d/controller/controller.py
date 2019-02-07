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

        self._set_to_default_values()
        self._setup_bg_tasks_pipeline()
        plugin.add_observer("recent_events", self._on_recent_events)

    def _set_to_default_values(self):
        self._model_being_updated = False

    def _on_recent_events(self, events):
        if "frame" in events:
            self._get_observation(events["frame"])
        self._update_model()

    def _get_observation(self, frame):
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
        self._controller_storage.save_observation(
            marker_id_to_detections, camera_extrinsics
        )

        self._visibility_graphs.check_novel_markers(
            marker_id_to_detections, self._controller_storage.current_frame_id
        )

    def _update_model(self):
        if self._model_being_updated:
            return

        data_for_model_init = self._prepare_for_model_update.run()
        if not data_for_model_init:
            return

        self._model_being_updated = True
        self._model_initialization_controller.run(data_for_model_init)  # bg_task

    # TODO: merge ModelInitializationController and ModelOptimizationController;
    #  remove observers here
    def _setup_bg_tasks_pipeline(self):
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

    def _on_update_model_finished(self):
        self._model_being_updated = False

    def reset(self):
        self._set_to_default_values()
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
