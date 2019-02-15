import logging

import camera_models
from marker_tracker_3d import controller

logger = logging.getLogger(__name__)


class GeneralController:
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

        self._observation_process_controller = controller.ObservationProcessController(
            controller_storage, model_storage, camera_intrinsics
        )
        self._model_update_controller = controller.ModelUpdateController(
            controller_storage, model_storage, camera_intrinsics, task_manager
        )

        plugin.add_observer("recent_events", self._on_recent_events)

    def _on_recent_events(self, events):
        if "frame" in events:
            self._observation_process_controller.run(events["frame"])
            self._controller_storage.current_frame_id += 1

        if self._model_storage.optimize_model_allowed:
            self._model_update_controller.run()

    def reset(self):
        self._controller_storage.reset()
        self._model_storage.reset()
        self._observation_process_controller.reset()
        self._model_update_controller.reset()
        logger.info("Reset 3D Marker Tracker!")

    def load_marker_tracker_3d_model(self):
        self._model_storage.load_marker_tracker_3d_model_from_file()

    def export_marker_tracker_3d_model(self):
        self._model_storage.export_marker_tracker_3d_model()

    # TODO: debug only; to be removed
    def export_visibility_graph(self):
        self._model_storage.export_visibility_graph(
            self._controller_storage.current_frame_id
        )

    def optimize_camera_intrinsics_switch(self, optimize_camera_intrinsics):
        self._model_update_controller.optimize_camera_intrinsics_switch(
            optimize_camera_intrinsics
        )

    # TODO: maybe should be moved to other place
    def load_camera_intrinsics(self):
        camera_intrinsics = camera_models.load_intrinsics(
            self._save_path,
            self._camera_intrinsics.name,
            self._camera_intrinsics.resolution,
        )
        self._camera_intrinsics.update_camera_matrix(camera_intrinsics.K)
        self._camera_intrinsics.update_dist_coefs(camera_intrinsics.D)

    # TODO: maybe should be moved to other place
    def export_camera_intrinsics(self):
        self._camera_intrinsics.save(self._save_path)

    def export_all_camera_poses(self):
        self._controller_storage.export_all_camera_poses()
