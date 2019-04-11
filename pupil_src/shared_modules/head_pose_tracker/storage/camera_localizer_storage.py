"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os

import file_methods as fm
import player_methods as pm
from observable import Observable


class CameraLocalizerStorage(Observable):
    def __init__(self, rec_dir, plugin):
        self._rec_dir = rec_dir

        self.pose_bisector = pm.Mutable_Bisector()

        self.load_pldata_from_disk()

        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_cleanup(self):
        self.save_pldata_to_disk()

    @property
    def calculated(self):
        return bool(self.pose_bisector)

    def save_pldata_to_disk(self):
        self._save_to_file()

    def _save_to_file(self):
        directory = self._offline_data_folder_path
        file_name = self._pldata_file_name
        os.makedirs(directory, exist_ok=True)
        with fm.PLData_Writer(directory, file_name) as writer:
            for pose_ts, pose in zip(
                self.pose_bisector.timestamps, self.pose_bisector.data
            ):
                writer.append_serialized(
                    pose_ts, topic="pose", datum_serialized=pose.serialized
                )

    def load_pldata_from_disk(self):
        self._load_from_file()

    def _load_from_file(self):
        directory = self._offline_data_folder_path
        file_name = self._pldata_file_name
        pldata = fm.load_pldata_file(directory, file_name)
        self.pose_bisector = pm.Mutable_Bisector(pldata.data, pldata.timestamps)

    @property
    def _offline_data_folder_path(self):
        return os.path.join(self._rec_dir, "offline_data")

    @property
    def _pldata_file_name(self):
        return "camera_poses"
