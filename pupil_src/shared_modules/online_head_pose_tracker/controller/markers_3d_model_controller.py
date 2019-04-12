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
from observable import Observable
from online_head_pose_tracker import worker, storage

logger = logging.getLogger(__name__)


class Markers3DModelController(Observable):
    def __init__(
        self,
        general_settings,
        marker_location_storage,
        markers_3d_model_storage,
        camera_intrinsics,
        task_manager,
        user_dir,
    ):
        self._general_settings = general_settings
        self._marker_location_storage = marker_location_storage
        self._markers_3d_model_storage = markers_3d_model_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._user_dir = user_dir

        self._optimization_storage = storage.OptimizationStorage()
        self._pick_key_markers = worker.PickKeyMarkers(
            self._optimization_storage, select_key_markers_interval=1
        )

        self._task = None

    def pick_key_markers(self):
        self._pick_key_markers.run(self._marker_location_storage.current_markers)

    def calculate(self):
        if self._task is None or not self._task.running:
            if self._check_ready():
                self._create_optimize_markers_3d_model_task()

    def _check_ready(self):
        try:
            self._optimization_storage.marker_id_to_extrinsics_opt[
                self._optimization_storage.origin_marker_id
            ]
        except KeyError:
            self._optimization_storage.set_origin_marker_id()
        return len(self._optimization_storage.all_key_markers) > 50

    def _reset(self):
        if self._task is not None and self._task.running:
            self._task.kill(None)

    def _create_optimize_markers_3d_model_task(self):
        def on_completed(result):
            self._update_result(result)

            self._markers_3d_model_storage.save_plmodel_to_disk()

        self._task = worker.optimize_markers_3d_model.create_task(
            self._optimization_storage, self._general_settings, self._camera_intrinsics
        )
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer(
            "on_started", self.on_markers_3d_model_optimization_started
        )
        self._task_manager.add_task(self._task)

    def _update_result(self, result):
        if not result:
            return
        bundle_adjustment_result, intrinsics = result
        worker.update_optimization_storage.run(
            self._optimization_storage, bundle_adjustment_result
        )

        model_data = {
            "marker_id_to_extrinsics": self._optimization_storage.marker_id_to_extrinsics_opt,
            "marker_id_to_points_3d": self._optimization_storage.marker_id_to_points_3d_opt,
            "origin_marker_id": self._optimization_storage.origin_marker_id,
            "centroid": self._optimization_storage.centroid,
        }
        self._markers_3d_model_storage.model = model_data
        self._camera_intrinsics.update_camera_matrix(intrinsics["camera_matrix"])
        self._camera_intrinsics.update_dist_coefs(intrinsics["dist_coefs"])

    def on_markers_3d_model_optimization_started(self):
        pass
