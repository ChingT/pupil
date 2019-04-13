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


class Settings:
    version = 1

    def __init__(self):
        self.set_to_default_values()

    def set_to_default_values(self):
        self.optimize_camera_intrinsics = False
        self.show_marker_id = False
        self.show_camera_trace = True

    def _load_settings(
        self, optimize_camera_intrinsics, show_marker_id, show_camera_trace
    ):
        self.optimize_camera_intrinsics = optimize_camera_intrinsics
        self.show_marker_id = show_marker_id
        self.show_camera_trace = show_camera_trace

    @property
    def as_tuple(self):
        return (
            self.optimize_camera_intrinsics,
            self.show_marker_id,
            self.show_camera_trace,
        )


class OnlineGeneralSettings(Settings):
    def __init__(self, user_dir, plugin):
        super().__init__()

        self._user_dir = user_dir

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
            try:
                self._load_settings(*settings_tuple)
            except TypeError:
                pass

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
    def _msgpack_file_name(self):
        return "head_pose_tracker_general_settings.msgpack"

    @property
    def _msgpack_file_path(self):
        return os.path.join(self._user_dir, self._msgpack_file_name)
