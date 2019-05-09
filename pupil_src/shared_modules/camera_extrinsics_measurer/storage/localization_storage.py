"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import os

import numpy as np

import file_methods as fm
import player_methods as pm
from camera_extrinsics_measurer.function import utils
from observable import Observable


class Localization:
    def __init__(self):
        self.set_to_default_values()

    def set_to_default_values(self):
        self.recent_camera_trace = collections.deque(maxlen=300)

    def add_recent_camera_trace(self, camera_trace):
        self.recent_camera_trace.append(camera_trace)

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
        super().__init__()

        self._get_current_frame_window = get_current_frame_window

    def set_to_default_values(self):
        super().set_to_default_values()
        self.pose_bisector = {
            name: pm.Mutable_Bisector() for name in ["world", "eye0", "eye1"]
        }

    @property
    def calculated(self):
        return bool(self.pose_bisector["world"])

    @property
    def current_pose(self):
        frame_window = self._get_current_frame_window()
        current_poses = {
            camera_name: self.none_pose_data
            for camera_name in ["world", "eye0", "eye1"]
        }
        try:
            pose_data_world = self.pose_bisector["world"].by_ts_window(frame_window)[0]
        except IndexError:
            return current_poses
        else:
            current_poses["world"].update(
                {
                    "camera_extrinsics": np.zeros((6,), dtype=np.float32).tolist(),
                    "camera_poses": np.zeros((6,), dtype=np.float32).tolist(),
                    "camera_trace": np.zeros((3,), dtype=np.float32).tolist(),
                    "camera_pose_matrix": np.eye(4, dtype=np.float32).tolist(),
                }
            )
            inv = utils.convert_extrinsic_to_matrix(
                pose_data_world["camera_extrinsics"]
            )

        for camera_name in ["eye0", "eye1"]:
            try:
                pose_data = self.pose_bisector[camera_name].by_ts_window(frame_window)[
                    0
                ]
            except IndexError:
                pass
            else:
                camera_pose_matrix = inv @ pose_data["camera_pose_matrix"]
                camera_poses = utils.convert_matrix_to_extrinsic(camera_pose_matrix)
                camera_extrinsics = utils.get_camera_pose(camera_poses)
                current_poses[camera_name].update(
                    {
                        "camera_extrinsics": camera_extrinsics.tolist(),
                        "camera_poses": camera_poses.tolist(),
                        "camera_trace": camera_poses[3:6].tolist(),
                        "camera_pose_matrix": camera_pose_matrix.tolist(),
                    }
                )

        return current_poses


class OfflineLocalizationStorage(Observable, OfflineCameraLocalization):
    def __init__(self, rec_dir, plugin, get_current_frame_window):
        super().__init__(get_current_frame_window)

        self._rec_dir = rec_dir

        self.load_pldata_from_disk()

        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_cleanup(self):
        self.save_pldata_to_disk()

    def save_pldata_to_disk(self):
        self._save_to_file()

    def _save_to_file(self):
        file_name = self._pldata_file_name
        for camera_name in ["world", "eye0", "eye1"]:
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
        for camera_name in ["world", "eye0", "eye1"]:
            directory = self._offline_data_folder_path(camera_name)
            pldata = fm.load_pldata_file(directory, file_name)
            self.pose_bisector[camera_name] = pm.Mutable_Bisector(
                pldata.data, pldata.timestamps
            )

    def _offline_data_folder_path(self, camera_name):
        return os.path.join(self._rec_dir, "offline_data", camera_name)

    @property
    def _pldata_file_name(self):
        return "camera_pose"


class OnlineLocalizationStorage(Localization):
    def __init__(self):
        super().__init__()

        self.current_pose = self.none_pose_data
