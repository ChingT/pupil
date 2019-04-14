"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import tasklib
from head_pose_tracker import worker
from head_pose_tracker.function import pick_key_markers


class OnlineController:
    def __init__(
        self,
        general_settings,
        marker_location_storage,
        markers_3d_model_storage,
        camera_localizer_storage,
        camera_intrinsics,
        task_manager,
        user_dir,
        plugin,
    ):
        self._general_settings = general_settings
        self._marker_location_storage = marker_location_storage
        self._markers_3d_model_storage = markers_3d_model_storage
        self._camera_localizer_storage = camera_localizer_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._user_dir = user_dir

        self._task = None

        # first trigger
        self._calculate_markers_3d_model()

        plugin.add_observer("recent_events", self._on_recent_events)
        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_recent_events(self, events):
        if "frame" in events:
            self._calculate_current_markers(events["frame"])
            self._calculate_current_pose()
            self._save_key_markers()
            self._calculate_markers_3d_model()

    def _calculate_current_markers(self, frame):
        self._marker_location_storage.current_markers = worker.online_detection(frame)

    def _calculate_current_pose(self):
        if not self._markers_3d_model_storage.calculated:
            return

        self._camera_localizer_storage.current_pose = worker.online_localization(
            self._marker_location_storage.current_markers,
            self._markers_3d_model_storage.marker_id_to_extrinsics,
            self._camera_localizer_storage,
            self._camera_intrinsics,
        )

    def _save_key_markers(self):
        self._markers_3d_model_storage.all_key_markers += pick_key_markers.run(
            self._marker_location_storage.current_markers,
            self._markers_3d_model_storage.all_key_markers,
        )

    def _calculate_markers_3d_model(self):
        def on_completed(result):
            if result:
                self._update_result(result)
                self._markers_3d_model_storage.save_plmodel_to_disk()
            # Start again the task when finished
            self._calculate_markers_3d_model()

        def on_canceled_or_killed():
            self._markers_3d_model_storage.save_plmodel_to_disk()

        if self.is_running_task:
            return
        self._task = self._create_task()
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_canceled_or_killed", on_canceled_or_killed)
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
        model_tuple, frame_id_to_extrinsics, frame_ids_failed, intrinsics_tuple = result
        self._markers_3d_model_storage.update_model(*model_tuple)

        self._markers_3d_model_storage.frame_id_to_extrinsics = frame_id_to_extrinsics
        self._markers_3d_model_storage.discard_failed_key_markers(frame_ids_failed)

        self._camera_intrinsics.update_camera_matrix(intrinsics_tuple.camera_matrix)
        self._camera_intrinsics.update_dist_coefs(intrinsics_tuple.dist_coefs)

    @property
    def is_running_task(self):
        return self._task is not None and self._task.running

    def cancel_task(self):
        if self.is_running_task:
            self._task.kill(None)

    def _on_cleanup(self):
        self._camera_intrinsics.save(self._user_dir)
