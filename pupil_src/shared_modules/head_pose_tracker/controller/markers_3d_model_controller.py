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
from observable import Observable


class Markers3DModelController(Observable):
    def __init__(
        self,
        markers_3d_model_storage,
        marker_location_storage,
        camera_intrinsics,
        task_manager,
        get_current_trim_mark_range,
        recording_uuid,
        rec_dir,
    ):
        self._markers_3d_model_storage = markers_3d_model_storage
        self._marker_location_storage = marker_location_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._recording_uuid = recording_uuid
        self._rec_dir = rec_dir
        self._task = None

    def calculate(self, markers_3d_model):
        def on_yield_markers_3d_model(result):
            markers_3d_model.status = (
                "Building Markers 3D Model {:.0f}% "
                "complete".format(self._task.progress * 100)
            )
            markers_3d_model_result, camera_intrinsics = result
            markers_3d_model.update_result(markers_3d_model_result)
            self._camera_intrinsics.update_camera_matrix(camera_intrinsics.K)
            self._camera_intrinsics.update_dist_coefs(camera_intrinsics.D)

        def on_completed_markers_3d_model(_):
            markers_3d_model.status = "Building Markers 3D Model successfully"
            self._markers_3d_model_storage.save_to_disk()
            self._camera_intrinsics.save(self._rec_dir)
            self.on_markers_3d_model_computed()

        self._reset(markers_3d_model)
        self._task = worker.create_markers_3d_model.create_task(
            markers_3d_model, self._marker_location_storage
        )
        self._task.add_observer("on_yield", on_yield_markers_3d_model)
        self._task.add_observer("on_completed", on_completed_markers_3d_model)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task_manager.add_task(self._task)
        return self._task

    def _reset(self, markers_3d_model):
        if self._task is not None and self._task.running:
            self._task.kill(None)
        markers_3d_model.reset()

    def on_markers_3d_model_computed(self):
        pass

    def set_markers_3d_model_range_from_current_trim_marks(self, markers_3d_model):
        markers_3d_model.frame_index_range = self._get_current_trim_mark_range()

    def is_from_same_recording(self, markers_3d_model):
        """
        False if the markers_3d_model file was copied from another recording directory
        """
        return (
            markers_3d_model is not None
            and markers_3d_model.recording_uuid == self._recording_uuid
        )
