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
import os

import file_methods as fm

logger = logging.getLogger(__name__)


class GeneralSettings:
    version = 1

    def __init__(self, rec_dir, get_recording_index_range, plugin):
        self._rec_dir = rec_dir
        self._get_recording_index_range = get_recording_index_range

        plugin.add_observer("cleanup", self._on_cleanup)

        self.load_from_disk()

    def _on_cleanup(self):
        self.save_to_disk()

    def save_to_disk(self):
        self._save_msgpack_to_file(self._msgpack_file_path, self.as_tuple)

    def _save_msgpack_to_file(self, file_path, data):
        dict_representation = {"version": self.version, "data": data}
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fm.save_object(dict_representation, file_path)

    def load_from_disk(self):
        settings_tuple = self._load_msgpack_from_file(self._msgpack_file_path)
        if settings_tuple:
            self._load_default_settings(settings_tuple)
        else:
            self._create_default_settings()

    @property
    def as_tuple(self):
        return (
            self.marker_location_frame_index_range,
            self.marker_location_calculated_frame_indices,
            self.markers_3d_model_frame_index_range,
            self.markers_3d_model_status,
            self.user_defined_origin_marker_id,
            self.optimize_camera_intrinsics,
            self.show_marker_id,
            self.camera_localizer_frame_index_range,
            self.camera_localizer_status,
            self.show_camera_trace,
        )

    def _load_default_settings(self, settings_tuple):
        (
            self.marker_location_frame_index_range,
            self.marker_location_calculated_frame_indices,
            self.markers_3d_model_frame_index_range,
            self.markers_3d_model_status,
            self.user_defined_origin_marker_id,
            self.optimize_camera_intrinsics,
            self.show_marker_id,
            self.camera_localizer_frame_index_range,
            self.camera_localizer_status,
            self.show_camera_trace,
        ) = settings_tuple

    def _create_default_settings(self):
        self.marker_location_frame_index_range = self._get_recording_index_range()
        # TODO: fix marker_location_calculated_frame_indices
        self.marker_location_calculated_frame_indices = []
        self.markers_3d_model_frame_index_range = self._get_recording_index_range()
        self.markers_3d_model_status = "Not calculated yet"
        self.user_defined_origin_marker_id = None
        self.optimize_camera_intrinsics = False
        self.show_marker_id = False
        self.camera_localizer_frame_index_range = self._get_recording_index_range()
        self.camera_localizer_status = "Not calculated yet"
        self.show_camera_trace = True

    def _load_msgpack_from_file(self, file_path):
        try:
            dict_representation = fm.load_object(file_path)
        except FileNotFoundError:
            return None
        if dict_representation.get("version", None) != self.version:
            logger.warning(
                "Data in {} is in old file format. Will not load these!".format(
                    file_path
                )
            )
            return None
        return dict_representation.get("data", None)

    @property
    def _offline_data_folder_path(self):
        return os.path.join(self._rec_dir, "offline_data")

    @property
    def _msgpack_file_name(self):
        return "head_pose_tracker_general_settings.msgpack"

    @property
    def _msgpack_file_path(self):
        return os.path.join(self._offline_data_folder_path, self._msgpack_file_name)
