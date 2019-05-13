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

import numpy as np

import file_methods as fm
import player_methods as pm
from camera_extrinsics_measurer.function import utils
from observable import Observable


class Localization:
    @property
    def none_pose_data(self):
        return {
            "camera_extrinsics": None,
            "camera_poses": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "camera_trace": [np.nan, np.nan, np.nan],
            "camera_pose_matrix": None,
        }


class OfflineCameraLocalization(Localization):
    def __init__(self, get_current_frame_window):
        self.pose_bisector = {name: pm.Mutable_Bisector() for name in utils.camera_name}
        self._get_current_frame_window = get_current_frame_window

    def set_to_default_values(self, camera_name):
        self.pose_bisector[camera_name] = pm.Mutable_Bisector()

    def calculated(self, camera_name):
        return bool(self.pose_bisector[camera_name])

    @property
    def current_pose(self):
        frame_window = self._get_current_frame_window()
        current_poses = {
            camera_name: self.none_pose_data for camera_name in utils.camera_name
        }

        for camera_name in utils.camera_name:
            pose_datum = self.pose_bisector[camera_name].by_ts_window(frame_window)
            try:
                pose_data = pose_datum[0]
            except IndexError:
                pass
            else:
                current_poses[camera_name] = pose_data

        return current_poses


class OfflineLocalizationStorage(Observable, OfflineCameraLocalization):
    def __init__(self, rec_dir, get_current_frame_window):
        super().__init__(get_current_frame_window)

        self._rec_dir = rec_dir

        self.load_pldata_from_disk()

    def save_pldata_to_disk(self):
        self._save_to_file()

    def _save_to_file(self):
        file_name = self._pldata_file_name
        for camera_name in utils.camera_name:
            directory = self._offline_data_folder_path(camera_name)
            os.makedirs(directory, exist_ok=True)
            with fm.PLData_Writer(directory, file_name) as writer:
                for pose_ts, pose in zip(
                    self.pose_bisector[camera_name].timestamps,
                    self.pose_bisector[camera_name].data,
                ):
                    writer.append_serialized(
                        pose_ts, topic="pose", datum_serialized=pose.serialized
                    )

    def load_pldata_from_disk(self):
        self._load_from_file()

    def _load_from_file(self):
        file_name = self._pldata_file_name
        for camera_name in utils.camera_name:
            directory = self._offline_data_folder_path(camera_name)
            pldata = fm.load_pldata_file(directory, file_name)
            self.pose_bisector[camera_name] = pm.Mutable_Bisector(
                pldata.data, pldata.timestamps
            )

    @property
    def _pldata_file_name(self):
        return "camera_pose"

    def _offline_data_folder_path(self, camera_name):
        return os.path.join(self._rec_dir, "offline_data", camera_name)
