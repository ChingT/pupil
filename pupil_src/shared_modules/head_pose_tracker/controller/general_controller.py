"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

import video_capture
from head_pose_tracker import controller

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
        self._plugin = plugin

        self._observation_process_controller = controller.ObservationProcessController(
            controller_storage, model_storage, camera_intrinsics
        )
        self._model_update_controller = controller.ModelUpdateController(
            controller_storage, model_storage, camera_intrinsics, task_manager
        )

        plugin.add_observer("recent_events", self._on_recent_events)
        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_recent_events(self, events):
        if "frame" in events:
            self._observation_process_controller.run(events["frame"])

        if self._model_storage.optimize_3d_model:
            self._model_update_controller.run()

    def reset(self):
        logger.info("Reset Markers 3D Model!")
        self._controller_storage.reset()
        self._model_storage.reset()
        self._observation_process_controller.reset()
        self._model_update_controller.reset()

    def export_markers_3d_model_to_file(self):
        self._model_storage.export_markers_3d_model_to_file()

    def export_all_camera_extrinsics(self):
        self._controller_storage.export_all_camera_extrinsics()

    # TODO: maybe should be moved to other place
    def export_camera_intrinsics(self):
        self._camera_intrinsics.save(self._save_path)

    # TODO: debug only; to be removed
    def export_visibility_graph(self):
        self._model_storage.export_visibility_graph()

    def _on_cleanup(self):
        self.export_markers_3d_model_to_file()
        self.export_all_camera_extrinsics()
        self.export_camera_intrinsics()


class OfflineGeneralController:
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
        self._task_manager = task_manager
        self._save_path = save_path
        self._plugin = plugin

        self._observation_process_controller = controller.ObservationProcessController(
            controller_storage, model_storage, camera_intrinsics
        )
        self._model_update_controller = controller.ModelUpdateController(
            controller_storage, model_storage, camera_intrinsics, task_manager
        )

        plugin.add_observer("recent_events", self._on_recent_events)
        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_recent_events(self, events):
        try:
            current_frame_id = events["frame"].timestamp
        except KeyError:
            return

        try:
            marker_id_to_detections = self._controller_storage.all_marker_id_to_detections[
                current_frame_id
            ]
        except KeyError:
            pass
        else:
            self._controller_storage.update_current_marker_id_to_detections(
                marker_id_to_detections
            )

        try:
            camera_extrinsics = self._controller_storage.all_camera_extrinsics[
                current_frame_id
            ]
        except KeyError:
            pass
        else:
            self._controller_storage.update_current_camera_pose(camera_extrinsics)

    def start_optimize(self):
        if not self._controller_storage.all_marker_id_to_detections:
            self._detect_all_images()
        if not self._controller_storage.all_key_markers:
            self._pick_keyframes()
        self._add_from_key_edges_queue()
        self._model_update_controller.run()
        # self.export_markers_3d_model_to_file()

    def _detect_all_images(self):
        cap = video_capture.File_Source(
            self._plugin.g_pool,
            source_path=self._plugin.g_pool.capture.source_path,
            timing=None,
        )
        while True:
            try:
                frame = cap.get_frame()
                current_frame_id = frame.timestamp
                self._observation_process_controller.get_marker_id_to_detections(
                    frame, current_frame_id
                )
            except video_capture.EndofVideoError:
                self._controller_storage.export_all_marker_id_to_detections()
                return

    def _pick_keyframes(self):
        for (
            frame_id,
            marker_id_to_detections,
        ) in self._controller_storage.all_marker_id_to_detections.items():
            self._observation_process_controller.pick_key_markers(
                marker_id_to_detections, frame_id
            )
        self.export_all_key_markers_and_edges()

    def start_localize(self):
        if not self._controller_storage.all_marker_id_to_detections:
            self._detect_all_images()
        self._localize()
        self.export_all_camera_extrinsics()

    def _localize(self):
        for (
            frame_id,
            marker_id_to_detections,
        ) in self._controller_storage.all_marker_id_to_detections.items():
            self._observation_process_controller.localize(
                marker_id_to_detections, frame_id
            )

    def _add_from_key_edges_queue(self):
        self._model_storage.visibility_graph.add_edges_from(
            self._controller_storage.key_edges_queue
        )
        del self._controller_storage.key_edges_queue[:]

    def reset(self):
        logger.info("Reset Markers 3D Model!")
        self._controller_storage.reset()
        self._model_storage.reset()
        self._observation_process_controller.reset()
        self._model_update_controller.reset()

    def export_markers_3d_model_to_file(self):
        self._model_storage.export_markers_3d_model_to_file()

    def export_all_camera_extrinsics(self):
        self._controller_storage.export_all_camera_extrinsics()

    def export_all_key_markers_and_edges(self):
        self._controller_storage.export_all_key_markers()
        self._controller_storage.export_all_key_edges()

    # TODO: maybe should be moved to other place
    def export_camera_intrinsics(self):
        self._camera_intrinsics.save(self._save_path)

    # TODO: debug only; to be removed
    def export_visibility_graph(self):
        self._model_storage.export_visibility_graph()

    def _on_cleanup(self):
        self.export_markers_3d_model_to_file()
        self.export_camera_intrinsics()
