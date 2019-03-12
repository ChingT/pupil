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

    def __init__(self, g_pool, predetermined_origin_marker_id=22):
        super().__init__(g_pool)

        self.inject_plugin_dependencies()

        self._task_manager = PluginTaskManager(plugin=self)

        self._recording_uuid = self._load_recording_uuid_from_info_csv()

        self._setup_storages(predetermined_origin_marker_id)
        self._setup_controllers()
        self._setup_ui()
        self._setup_timelines()

        self._calculate_all_controller.calculate_all()

    def _setup_storages(self, predetermined_origin_marker_id):
        self._controller_storage = model.ControllerStorage(
            save_path=self.g_pool.rec_dir
        )
        self._model_storage = model.ModelStorage(
            predetermined_origin_marker_id, save_path=self.g_pool.rec_dir
        )

        self._marker_location_storage = model.MarkerLocationStorage(
            self.g_pool.rec_dir, plugin=self
        )

        self._optimization_storage = model.OptimizationStorage(
            rec_dir=self.g_pool.rec_dir,
            plugin=self,
            get_recording_index_range=self._recording_index_range,
            recording_uuid=self._recording_uuid,
            model_storage=self._model_storage,
        )
        self._camera_localizer_storage = model.CameraLocalizerStorage(
            self._optimization_storage,
            rec_dir=self.g_pool.rec_dir,
            plugin=self,
            get_recording_index_range=self._recording_index_range,
        )

    def _setup_controllers(self):
        self._marker_detection_controller = controller.MarkerDetectionController(
            self._task_manager, self._marker_location_storage
        )

        self._optimization_controller = controller.OptimizationController(
            self._controller_storage,
            self._model_storage,
            self._optimization_storage,
            self._marker_location_storage,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
            recording_uuid=self._recording_uuid,
        )

        self._camera_localizer_controller = controller.CameraLocalizerController(
            self._camera_localizer_storage,
            self._optimization_storage,
            self._marker_location_storage,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
        )

        self._calculate_all_controller = controller.CalculateAllController(
            self._marker_detection_controller,
            self._marker_location_storage,
            self._optimization_controller,
            self._optimization_storage,
            self._camera_localizer_controller,
            self._camera_localizer_storage,
        )

    def _setup_ui(self):
        self._head_pose_tracker_menu = plugin_ui.OfflineHeadPoseTrackerMenu(
            self._controller_storage, self._model_storage, plugin=self
        )
        self.visualization_3d_window = plugin_ui.Visualization3dWindow(
            self.g_pool.capture.intrinsics,
            self._controller_storage,
            self._model_storage,
            plugin=self,
        )
        self._marker_renderer = plugin_ui.MarkerRenderer(
            self._controller_storage, self._model_storage, plugin=self
        )

    def _setup_timelines(self):
        self._marker_location_timeline = plugin_ui.MarkerLocationTimeline(
            self._marker_detection_controller, self._marker_location_storage
        )
        self._camera_localizer_timeline = plugin_ui.CameraLocalizerTimeline(
            self._camera_localizer_storage,
            self._camera_localizer_controller,
            self._optimization_storage,
            self._optimization_controller,
        )
        plugin_timeline = PluginTimeline(
            plugin=self,
            title="Offline Optimization",
            timeline_ui_parent=self.g_pool.user_timelines,
            all_timestamps=self.g_pool.timestamps,
        )
        self._timeline = plugin_ui.OfflineHeadPoseTrackerTimeline(
            plugin_timeline,
            self._marker_location_timeline,
            self._camera_localizer_timeline,
            plugin=self,
        )

    def inject_plugin_dependencies(self):
        from head_pose_tracker.worker.detect_square_markers import (
            SquareMarkerDetectionTask,
        )

        SquareMarkerDetectionTask.zmq_ctx = self.g_pool.zmq_ctx
        SquareMarkerDetectionTask.capture_source_path = self.g_pool.capture.source_path
        SquareMarkerDetectionTask.notify_all = self.notify_all

        from head_pose_tracker.worker import create_optimization

        create_optimization.g_pool = self.g_pool

        from head_pose_tracker.worker import map_gaze

        map_gaze.g_pool = self.g_pool

    def _seek_to_frame(self, frame_index):
        self.notify_all({"subject": "seek_control.should_seek", "index": frame_index})

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
