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
from head_pose_tracker import ui as plugin_ui, controller, model
from observable import Observable
from plugin import Plugin
from plugin_timeline import PluginTimeline
from tasklib.manager import PluginTaskManager


class Offline_Head_Pose_Tracker(Plugin, Observable):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the
    environment.
    """

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self._inject_plugin_dependencies()
        self._task_manager = PluginTaskManager(plugin=self)
        self._recording_uuid = self._load_recording_uuid_from_info_csv()

        self._setup_storages()
        self._setup_controllers()
        self._setup_menus()
        self._setup_renderers()
        self._setup_timelines()

        self._marker_location_controller.init_detection()

    def _setup_storages(self):
        self._marker_location_storage = model.MarkerLocationStorage(
            self.g_pool.rec_dir,
            plugin=self,
            get_recording_index_range=self._recording_index_range,
        )

        self._markers_3d_model_storage = model.Markers3DModelStorage(
            self.g_pool.rec_dir,
            plugin=self,
            get_recording_index_range=self._recording_index_range,
            recording_uuid=self._recording_uuid,
        )
        self._camera_localizer_storage = model.CameraLocalizerStorage(
            self.g_pool.rec_dir,
            plugin=self,
            get_recording_index_range=self._recording_index_range,
        )

    def _setup_controllers(self):
        self._marker_location_controller = controller.MarkerLocationController(
            self._marker_location_storage,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
        )
        self._markers_3d_model_controller = controller.Markers3DModelController(
            self._marker_location_controller,
            self._marker_location_storage,
            self._markers_3d_model_storage,
            self.g_pool.capture.intrinsics,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
            recording_uuid=self._recording_uuid,
            rec_dir=self.g_pool.rec_dir,
        )
        self._camera_localizer_controller = controller.CameraLocalizerController(
            self._markers_3d_model_controller,
            self._marker_location_storage,
            self._markers_3d_model_storage,
            self._camera_localizer_storage,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
        )

    def _setup_menus(self):
        self._marker_location_menu = plugin_ui.MarkerLocationMenu(
            self._marker_location_controller,
            self._marker_location_storage,
            index_range_as_str=self._index_range_as_str,
        )
        self._markers_3d_model_menu = plugin_ui.Markers3DModelMenu(
            self._markers_3d_model_storage,
            self._markers_3d_model_controller,
            index_range_as_str=self._index_range_as_str,
        )
        self._camera_localizer_menu = plugin_ui.CameraLocalizerMenu(
            self._camera_localizer_controller,
            self._camera_localizer_storage,
            index_range_as_str=self._index_range_as_str,
        )
        self._offline_head_pose_tracker_menu = plugin_ui.OfflineHeadPoseTrackerMenu(
            self._marker_location_menu,
            self._markers_3d_model_menu,
            self._camera_localizer_menu,
            plugin=self,
        )

    def _setup_renderers(self):
        self._marker_location_renderer = plugin_ui.MarkerLocationRenderer(
            self._marker_location_storage,
            self._markers_3d_model_storage,
            plugin=self,
            get_current_frame_index=self.g_pool.capture.get_frame_index,
        )
        self._head_pose_tracker_renderer = plugin_ui.HeadPoseTrackerRenderer(
            self._marker_location_storage,
            self._markers_3d_model_storage,
            self._camera_localizer_storage,
            self.g_pool.capture.intrinsics,
            plugin=self,
            get_current_frame_index=self.g_pool.capture.get_frame_index,
        )

    def _setup_timelines(self):
        self._marker_location_timeline = plugin_ui.MarkerLocationTimeline(
            self._marker_location_controller, self._marker_location_storage
        )
        self._camera_localizer_timeline = plugin_ui.CameraLocalizerTimeline(
            self._camera_localizer_controller, self._camera_localizer_storage
        )
        plugin_timeline = PluginTimeline(
            plugin=self,
            title="Offline Head Pose Tracker",
            timeline_ui_parent=self.g_pool.user_timelines,
            all_timestamps=self.g_pool.timestamps,
        )
        self._timeline = plugin_ui.OfflineHeadPoseTrackerTimeline(
            plugin_timeline,
            self._marker_location_timeline,
            self._camera_localizer_timeline,
        )

    def _inject_plugin_dependencies(self):
        from head_pose_tracker.worker import (
            detect_square_markers,
            create_markers_3d_model,
            localize_pose,
        )

        detect_square_markers.g_pool = self.g_pool
        create_markers_3d_model.g_pool = self.g_pool
        localize_pose.g_pool = self.g_pool

    def _recording_index_range(self):
        left_index = 0
        right_index = len(self.g_pool.timestamps) - 1
        return left_index, right_index

    def _current_trim_mark_range(self):
        right_idx = self.g_pool.seek_control.trim_right
        left_idx = self.g_pool.seek_control.trim_left
        return left_idx, right_idx

    def _index_range_as_str(self, index_range):
        from_index, to_index = index_range
        return "{} - {}".format(
            self._index_time_as_str(from_index), self._index_time_as_str(to_index)
        )

    def _index_time_as_str(self, index):
        ts = self.g_pool.timestamps[index]
        min_ts = self.g_pool.timestamps[0]
        time = ts - min_ts
        minutes = abs(time // 60)  # abs because it's sometimes -0
        seconds = round(time % 60)
        return "{:02.0f}:{:02.0f}".format(minutes, seconds)

    def _load_recording_uuid_from_info_csv(self):
        info_csv_path = os.path.join(self.g_pool.rec_dir, "info.csv")
        with open(info_csv_path, "r", encoding="utf-8") as csv_file:
            recording_info = csv_utils.read_key_value_file(csv_file)
            return recording_info["Recording UUID"]
