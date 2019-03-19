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
from observable import Observable

logger = logging.getLogger(__name__)


class Markers3DModelController(Observable):
    def __init__(
        self,
        marker_location_storage,
        markers_3d_model_storage,
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
        self._reset(markers_3d_model)
        self._create_optimize_markers_3d_model_task(markers_3d_model)

    def _reset(self, markers_3d_model):
        if self._task is not None and self._task.running:
            self._task.kill(None)

        markers_3d_model.status = "Not calculated yet"
        markers_3d_model.result = None

    def _create_optimize_markers_3d_model_task(self, markers_3d_model):
        def on_yield_markers_3d_model(result):
            markers_3d_model.status = (
                "Building markers 3d model {:.0f}% "
                "complete".format(self._task.progress * 100)
            )
            self._update_result(markers_3d_model, result)

        def on_completed_markers_3d_model(_):
            markers_3d_model.status = "Building markers 3d model successfully"
            self._markers_3d_model_storage.save_to_disk()
            self._camera_intrinsics.save(self._rec_dir)
            logger.info(
                "Complete building markers 3d model for '{}'".format(
                    markers_3d_model.name
                )
            )

            self.on_markers_3d_model_computed()

        self._task = worker.create_markers_3d_model.create_task(
            markers_3d_model, self._marker_location_storage
        )
        self._task.add_observer("on_yield", on_yield_markers_3d_model)
        self._task.add_observer("on_completed", on_completed_markers_3d_model)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task_manager.add_task(self._task)
        logger.info(
            "Start building markers 3d model for '{}'".format(markers_3d_model.name)
        )
        self.on_markers_3d_model_calculating()

    def _update_result(self, markers_3d_model, result):
        model_datum, camera_intrinsics = result

        markers_3d_model.result = model_datum

        self._camera_intrinsics.update_camera_matrix(camera_intrinsics.K)
        self._camera_intrinsics.update_dist_coefs(camera_intrinsics.D)

    def on_markers_3d_model_calculating(self):
        pass

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
