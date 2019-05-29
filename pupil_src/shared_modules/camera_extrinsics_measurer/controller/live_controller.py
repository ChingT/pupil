"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from camera_extrinsics_measurer import worker
from camera_extrinsics_measurer.function import pick_key_markers


class LiveController:
    def __init__(
        self,
        general_settings,
        detection_storage,
        optimization_storage,
        localization_storage,
        intrinsics_dict,
        user_dir,
    ):
        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._optimization_storage = optimization_storage
        self._localization_storage = localization_storage
        self._intrinsics_dict = intrinsics_dict
        self._user_dir = user_dir

        # first trigger

    def recent_events(self, frame, camera_name):
        self._calculate_current_markers(frame, camera_name)
        self._calculate_current_pose(frame, camera_name)
        # self._save_key_markers(camera_name)

    def _calculate_current_markers(self, frame, camera_name):
        self._detection_storage.current_markers[camera_name] = worker.online_detection(
            frame
        )

    def _calculate_current_pose(self, frame, camera_name):
        current_pose = worker.online_localization(
            camera_name,
            frame.timestamp,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            self._intrinsics_dict[camera_name],
        )
        if current_pose is not None:
            self._localization_storage.current_pose[camera_name] = current_pose

        self._localization_storage.current_pose_converted = worker.online_convert_to_cam_coordinate(
            self._localization_storage.current_pose
        )

    def _save_key_markers(self, camera_name):
        if self._general_settings.optimize_camera_intrinsics:
            self._optimization_storage.all_key_markers[
                camera_name
            ] += pick_key_markers.run(
                self._detection_storage.current_markers[camera_name],
                self._optimization_storage.all_key_markers[camera_name],
            )

    def calculate_markers_3d_model(self, camera_name):
        if self._general_settings.optimize_camera_intrinsics:
            self._create_optimization_task(camera_name)

    def _create_optimization_task(self, camera_name):
        self._create_task(camera_name)

    def _create_task(self, camera_name):
        args = (
            self._intrinsics_dict[camera_name],
            self._optimization_storage.origin_marker_id,
            self._optimization_storage.marker_id_to_extrinsics,
            self._optimization_storage.frame_id_to_extrinsics,
            self._optimization_storage.all_key_markers[camera_name],
            self._general_settings.optimize_camera_intrinsics,
        )
        result = worker.online_optimization(*args)
        self._update_result(camera_name, result)
        # self.calculate_markers_3d_model()

    def _update_result(self, camera_name, result):
        if not result:
            return

        model_tuple, frame_id_to_extrinsics, valid_key_marker_ids, intrinsics_tuple = (
            result
        )
        self._optimization_storage.update_model(*model_tuple)
        self._optimization_storage.frame_id_to_extrinsics = frame_id_to_extrinsics
        self._optimization_storage.filter_valid_key_marker_ids(valid_key_marker_ids)

        self._intrinsics_dict[camera_name].update_camera_matrix(
            intrinsics_tuple.camera_matrix
        )
        self._intrinsics_dict[camera_name].update_dist_coefs(
            intrinsics_tuple.dist_coefs
        )
        self._intrinsics_dict[camera_name].save(self._user_dir)

    def reset(self):
        self._optimization_storage.set_to_default_values()
        self._localization_storage.set_to_default_values()
        self.calculate_markers_3d_model()
