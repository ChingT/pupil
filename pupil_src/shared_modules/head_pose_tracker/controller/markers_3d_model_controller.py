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
        marker_location_controller,
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

        self._markers_3d_model = markers_3d_model_storage.item

        marker_location_controller.add_observer(
            "on_marker_detection_had_completed_before",
            self._on_marker_detection_had_completed_before,
        )
        marker_location_controller.add_observer(
            "on_marker_detection_ended", self._on_marker_detection_ended
        )

    def _on_marker_detection_had_completed_before(self):
        self.calculate(check_complete=True)

    def _on_marker_detection_ended(self):
        self.calculate(check_complete=True)

    def calculate(self, check_complete=False):
        if check_complete and self._markers_3d_model.result:
            self.on_building_markers_3d_model_had_completed_before()
        else:
            self._reset()
            self._create_optimize_markers_3d_model_task()

    def _reset(self):
        if self._task is not None and self._task.running:
            self._task.kill(None)

        self._markers_3d_model.status = "Not calculated yet"
        self._markers_3d_model.result = None

    def _create_optimize_markers_3d_model_task(self):
        def on_yield_markers_3d_model(result):
            self._markers_3d_model.status = (
                "Building markers 3d model {:.0f}% "
                "complete".format(self._task.progress * 100)
            )
            self._update_result(self._markers_3d_model, result)

        def on_completed_markers_3d_model(_):
            self._markers_3d_model.status = "Building markers 3d model successfully"
            self._markers_3d_model_storage.save_to_disk()
            self._camera_intrinsics.save(self._rec_dir)
            logger.info(
                "Complete building markers 3d model for '{}'".format(
                    self._markers_3d_model.name
                )
            )
            self.on_building_markers_3d_model_completed()

        self._task = worker.create_markers_3d_model.create_task(
            self._markers_3d_model, self._marker_location_storage
        )
        self._task.add_observer("on_yield", on_yield_markers_3d_model)
        self._task.add_observer("on_completed", on_completed_markers_3d_model)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer("on_started", self.on_building_markers_3d_model_started)
        self._task_manager.add_task(self._task)
        logger.info(
            "Start building markers 3d model for '{}'".format(
                self._markers_3d_model.name
            )
        )

    def _update_result(self, markers_3d_model, result):
        model_datum, intrinsics = result
        markers_3d_model.result = model_datum
        self._camera_intrinsics.update_camera_matrix(intrinsics["camera_matrix"])
        self._camera_intrinsics.update_dist_coefs(intrinsics["dist_coefs"])

    def on_building_markers_3d_model_had_completed_before(self):
        pass

    def on_building_markers_3d_model_started(self):
        pass

    def on_building_markers_3d_model_completed(self):
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
