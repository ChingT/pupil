from marker_tracker_3d import worker


class ObservationProcessController:
    def __init__(self, controller_storage, model_storage, camera_intrinsics):
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._camera_intrinsics = camera_intrinsics

        self._pick_key_markers = worker.PickKeyMarkers(model_storage)

    def run(self, frame):
        marker_id_to_detections = worker.detect_markers.detect(frame)

        camera_extrinsics = self.localize(marker_id_to_detections)
        self._controller_storage.save_observation(
            marker_id_to_detections, camera_extrinsics
        )

        if self._model_storage.optimize_model_allowed:
            key_markers = self._pick_key_markers.run(
                marker_id_to_detections, self._controller_storage.current_frame_id
            )
            self._model_storage.save_key_markers(
                key_markers, self._controller_storage.current_frame_id
            )

    def localize(self, marker_id_to_detections):
        return worker.localize_camera.localize(
            self._camera_intrinsics,
            marker_id_to_detections,
            self._model_storage.marker_id_to_extrinsics_opt,
            self._controller_storage.camera_extrinsics,
            self._model_storage.origin_marker_id,
        )

    def reset(self):
        self._pick_key_markers.reset()
