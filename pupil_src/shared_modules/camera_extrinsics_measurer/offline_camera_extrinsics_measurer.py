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

import csv_utils
import player_methods as pm
import video_capture
from camera_extrinsics_measurer import (
    ui as plugin_ui,
    controller,
    storage,
    camera_names,
)
from observable import Observable
from plugin import Plugin
from plugin_timeline import PluginTimeline
from tasklib.manager import PluginTaskManager


class Empty(object):
    pass


class Camera_Extrinsics_Measurer(Plugin, Observable):
    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self._rec_dir = self.g_pool.rec_dir

        self._task_manager = PluginTaskManager(plugin=self)
        self._current_recording_uuid = self._load_recording_uuid_from_info_csv()

        self._init_cameras()
        self._setup_classes()

        self._detection_controller.calculate("world")

    def _init_cameras(self):
        self._source_path_dict = {}
        self._intrinsics_dict = {}
        self._all_timestamps_dict = {}
        for camera_name in camera_names:
            source_path = os.path.join(self._rec_dir, "{}.mp4".format(camera_name))
            try:
                src = video_capture.File_Source(
                    Empty(),
                    timing="external",
                    source_path=source_path,
                    buffered_decoding=True,
                    fill_gaps=True,
                )
            except AssertionError:
                continue
            self._source_path_dict[camera_name] = source_path
            self._intrinsics_dict[camera_name] = src.intrinsics
            self._all_timestamps_dict[camera_name] = src.timestamps

    def _setup_classes(self):
        self._setup_storages()
        self._setup_controllers()
        self._setup_renderers()
        self._setup_menus()
        self._setup_timelines()

    def _setup_storages(self):
        self._offline_settings_storage = storage.OfflineSettingsStorage(
            self.g_pool.rec_dir,
            plugin=self,
            get_recording_ts_range=self._recording_ts_range,
        )
        self._detection_storage = storage.OfflineDetectionStorage(
            self.g_pool.rec_dir,
            all_timestamps_dict=self._all_timestamps_dict,
            plugin=self,
            get_current_frame_index=self.get_current_frame_index,
            get_current_frame_window=self.get_current_frame_window,
        )
        self._optimization_storage = storage.OptimizationStorage(self.g_pool.rec_dir)
        self._localization_storage = storage.OfflineLocalizationStorage(
            self.g_pool.rec_dir, get_current_frame_window=self.get_current_frame_window
        )

    def _setup_controllers(self):
        self._detection_controller = controller.OfflineDetectionController(
            self._offline_settings_storage,
            self._detection_storage,
            task_manager=self._task_manager,
            current_trim_mark_ts_range=self._current_trim_mark_ts_range,
            all_timestamps_dict=self._all_timestamps_dict,
            source_path_dict=self._source_path_dict,
        )
        self._optimization_controller = controller.OfflineOptimizationController(
            self._detection_controller,
            self._offline_settings_storage,
            self._detection_storage,
            self._optimization_storage,
            camera_intrinsics_dict=self._intrinsics_dict,
            task_manager=self._task_manager,
            current_trim_mark_ts_range=self._current_trim_mark_ts_range,
            all_timestamps_dict=self._all_timestamps_dict,
            rec_dir=self.g_pool.rec_dir,
        )
        self._localization_controller = controller.OfflineLocalizationController(
            self._detection_controller,
            self._optimization_controller,
            self._offline_settings_storage,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            camera_intrinsics_dict=self._intrinsics_dict,
            task_manager=self._task_manager,
            current_trim_mark_ts_range=self._current_trim_mark_ts_range,
            all_timestamps_dict=self._all_timestamps_dict,
            rec_dir=self.g_pool.rec_dir,
        )
        self._export_controller = controller.ExportController(
            self._optimization_storage,
            self._localization_storage,
            task_manager=self._task_manager,
            plugin=self,
        )

    def _setup_renderers(self):
        self._detection_renderer = plugin_ui.DetectionRenderer(
            self._offline_settings_storage,
            self._detection_storage,
            self._optimization_storage,
            plugin=self,
        )

        self._head_pose_tracker_3d_renderer = plugin_ui.HeadPoseTracker3DRenderer(
            self._offline_settings_storage,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            self._intrinsics_dict,
            plugin=self,
        )

    def _setup_menus(self):
        self._visualization_menu = plugin_ui.VisualizationMenu(
            self._offline_settings_storage, self._head_pose_tracker_3d_renderer
        )
        self._detection_menu = plugin_ui.OfflineDetectionMenu(
            self._detection_controller,
            self._offline_settings_storage,
            ts_range_as_str=self._ts_range_as_str,
        )
        self._optimization_menu = plugin_ui.OfflineOptimizationMenu(
            self._optimization_controller,
            self._offline_settings_storage,
            self._optimization_storage,
            ts_range_as_str=self._ts_range_as_str,
        )
        self._localization_menu = plugin_ui.OfflineLocalizationMenu(
            self._localization_controller,
            self._offline_settings_storage,
            self._localization_storage,
            ts_range_as_str=self._ts_range_as_str,
        )
        self._head_pose_tracker_menu = plugin_ui.OfflineHeadPoseTrackerMenu(
            self._visualization_menu,
            self._detection_menu,
            self._optimization_menu,
            self._localization_menu,
            plugin=self,
        )

    def _setup_timelines(self):
        self._detection_timeline = plugin_ui.DetectionTimeline(
            self._detection_controller,
            self._offline_settings_storage,
            self._detection_storage,
            all_timestamps=self._all_timestamps_dict["world"],
        )
        self._localization_timeline = plugin_ui.LocalizationTimeline(
            self._localization_controller,
            self._offline_settings_storage,
            self._localization_storage,
            all_timestamps=self._all_timestamps_dict["world"],
        )
        plugin_timeline = PluginTimeline(
            plugin=self,
            title="Camera Extrinsics Measurer",
            timeline_ui_parent=self.g_pool.user_timelines,
            all_timestamps=self._all_timestamps_dict["world"],
        )
        self._timeline = plugin_ui.OfflineHeadPoseTrackerTimeline(
            plugin_timeline,
            self._detection_timeline,
            self._localization_timeline,
            plugin=self,
        )

    def _recording_ts_range(self):
        return (
            self._all_timestamps_dict["world"][0],
            self._all_timestamps_dict["world"][-1],
        )

    def _current_trim_mark_ts_range(self):
        right_ts = self.g_pool.seek_control.trim_right_ts
        left_ts = self.g_pool.seek_control.trim_left_ts
        return left_ts, right_ts

    def _ts_range_as_str(self, ts_range):
        from_ts, to_ts = ts_range
        return "{} - {}".format(
            self._ts_time_as_str(from_ts), self._ts_time_as_str(to_ts)
        )

    def _ts_time_as_str(self, ts):
        min_ts = self._all_timestamps_dict["world"][0]
        time = ts - min_ts
        minutes = abs(time // 60)  # abs because it's sometimes -0
        seconds = round(time % 60)
        return "{:02.0f}:{:02.0f}".format(minutes, seconds)

    def _load_recording_uuid_from_info_csv(self):
        info_csv_path = os.path.join(self.g_pool.rec_dir, "info.csv")
        with open(info_csv_path, "r", encoding="utf-8") as csv_file:
            recording_info = csv_utils.read_key_value_file(csv_file)
            return recording_info["Recording UUID"]

    def get_current_frame_index(self):
        return self.g_pool.capture.get_frame_index()

    def get_current_frame_window(self):
        frame_index = self.get_current_frame_index()
        try:
            frame_window = pm.enclosing_window(
                self._all_timestamps_dict["world"], frame_index
            )
        except IndexError:
            frame_window = []
        return frame_window
