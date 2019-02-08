from marker_tracker_3d import worker


class ObservationProcessController:
    def __init__(self, controller_storage, model_storage, camera_intrinsics):
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._camera_intrinsics = camera_intrinsics

        self._visibility_graphs = worker.VisibilityGraphs(model_storage)

    def run(self, frame):
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

        if self._model_storage.optimize_model_allowed:
            self._visibility_graphs.check_key_markers(
                marker_id_to_detections, self._controller_storage.current_frame_id
            )
        self._controller_storage.current_frame_id += 1

    def reset(self):
        self._visibility_graphs.reset()
