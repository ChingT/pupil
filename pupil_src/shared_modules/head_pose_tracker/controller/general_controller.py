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

    def export_all_camera_poses(self):
        self._controller_storage.export_all_camera_poses()

    # TODO: maybe should be moved to other place
    def export_camera_intrinsics(self):
        self._camera_intrinsics.save(self._save_path)

    # TODO: debug only; to be removed
    def export_visibility_graph(self):
        self._model_storage.export_visibility_graph()

    def _on_cleanup(self):
        self.export_markers_3d_model_to_file()
        self.export_all_camera_poses()
        self.export_camera_intrinsics()
