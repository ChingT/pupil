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

from head_pose_tracker import controller
from observable import Observable

logger = logging.getLogger(__name__)


class OfflineGeneralController(Observable):
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

        self._model_update_controller = controller.ModelUpdateController(
            controller_storage, model_storage, camera_intrinsics, task_manager
        )

        source_path = plugin.g_pool.capture.source_path
        video_length = len(plugin.g_pool.timestamps)
        self._marker_detector_controller = controller.OfflineObservationController(
            controller_storage,
            model_storage,
            camera_intrinsics,
            task_manager,
            source_path,
            video_length,
        )

        plugin.add_observer("recent_events", self._on_recent_events)
        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_recent_events(self, events):
        try:
            frame = events["frame"]
        except KeyError:
            return

        self._marker_detector_controller.run(frame)

    def start_optimize(self):
        self._model_update_controller.update_model_storage.add_observer(
            "on_update_model_storage_done", self._optimize
        )

        self._optimize()

    def _optimize(self):
        self._model_update_controller.run()

        if self._controller_storage.n_key_markers_processed == len(
            self._controller_storage.all_key_markers
        ):
            self._model_update_controller.update_model_storage.remove_observer(
                "on_update_model_storage_done", self._optimize
            )
            self._controller_storage.n_key_markers_processed = 0

    def export_markers_3d_model_to_file(self):
        self._model_storage.export_markers_3d_model_to_file()

    def export_camera_extrinsics_cache(self):
        self._controller_storage.export_camera_extrinsics_cache()

    def export_all_key_markers_and_edges(self):
        self._controller_storage.export_all_key_markers()
        self._controller_storage.export_all_key_edges()

    def export_marker_cache(self):
        self._controller_storage.export_marker_cache()

    # TODO: maybe should be moved to other place
    def export_camera_intrinsics(self):
        self._camera_intrinsics.save(self._save_path)

    # TODO: debug only; to be removed
    def export_visibility_graph(self):
        self._controller_storage.export_visibility_graph(
            self._model_storage.origin_marker_id,
            self._model_storage.marker_id_to_extrinsics_opt.keys(),
            self._model_storage.marker_id_to_points_3d_init.keys(),
        )

    def _on_cleanup(self):
        self.export_marker_cache()
        self.export_camera_extrinsics_cache()
        # self.export_markers_3d_model_to_file()

    def reset(self):
        logger.info("Reset Markers 3D Model!")
        self._controller_storage.reset()
        self._model_storage.reset()
        self._observation_process_controller.reset()
        self._model_update_controller.reset()
