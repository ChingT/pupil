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

import tasklib
from head_pose_tracker import worker
from head_pose_tracker.function import pick_key_markers
from observable import Observable

logger = logging.getLogger(__name__)


class OnlineMarkers3DModelController(Observable):
    def __init__(
        self,
        general_settings,
        marker_location_storage,
        markers_3d_model_storage,
        camera_intrinsics,
        task_manager,
    ):
        self._general_settings = general_settings
        self._marker_location_storage = marker_location_storage
        self._markers_3d_model_storage = markers_3d_model_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager

        self._task = None

    def calculate(self):
        self._markers_3d_model_storage.all_key_markers += pick_key_markers.run(
            self._marker_location_storage.current_markers,
            self._markers_3d_model_storage.all_key_markers,
        )
        if not self.is_running_task:
            self._create_optimize_markers_3d_model_task()

    def _create_optimize_markers_3d_model_task(self):
        def on_completed(result):
            if result:
                self._update_result(result)
                self._markers_3d_model_storage.save_plmodel_to_disk()

        self._task = self._create_task()
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_exception", tasklib.raise_exception)

    def _create_task(self):
        args = (
            self._markers_3d_model_storage.origin_marker_id,
            self._markers_3d_model_storage.marker_id_to_extrinsics,
            self._markers_3d_model_storage.frame_id_to_extrinsics,
            self._markers_3d_model_storage.all_key_markers,
            self._general_settings.optimize_camera_intrinsics,
            self._camera_intrinsics,
        )
        return self._task_manager.create_background_task(
            name="markers 3d model optimization",
            routine_or_generator_function=worker.online_optimization,
            pass_shared_memory=False,
            args=args,
        )

    def _update_result(self, result):
        model_tuple, intrinsics_tuple, frame_ids_failed = result
        self._markers_3d_model_storage.discard_failed_key_markers(frame_ids_failed)
        self._markers_3d_model_storage.load_model(*model_tuple)
        self._camera_intrinsics.update_camera_matrix(intrinsics_tuple.camera_matrix)
        self._camera_intrinsics.update_dist_coefs(intrinsics_tuple.dist_coefs)

    @property
    def is_running_task(self):
        return self._task is not None and self._task.running

    def cancel_task(self):
        if self.is_running_task:
            self._task.kill(None)
