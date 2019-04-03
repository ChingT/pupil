"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from plugin_timeline import Row, RangeElementFrameIdx, BarsElementTs


class OfflineHeadPoseTrackerTimeline:
    def __init__(
        self, plugin_timeline, marker_location_timeline, camera_localizer_timeline
    ):
        self._plugin_timeline = plugin_timeline
        self._marker_location_timeline = marker_location_timeline
        self._camera_localizer_timeline = camera_localizer_timeline

        marker_location_timeline.render_parent_timeline = self.render
        camera_localizer_timeline.render_parent_timeline = self.render

    def render(self):
        self._plugin_timeline.clear_rows()
        self._plugin_timeline.add_row(self._marker_location_timeline.create_row())
        self._plugin_timeline.add_row(self._camera_localizer_timeline.create_row())
        self._plugin_timeline.refresh()


class MarkerLocationTimeline:
    timeline_label = "Marker detection"

    def __init__(self, marker_location_controller, marker_location_storage):
        self.render_parent_timeline = None

        self._marker_location_controller = marker_location_controller
        self._marker_locations = marker_location_storage.item

        marker_location_storage.add_observer("load_from_disk", self._on_storage_changed)
        marker_location_controller.add_observer(
            "on_marker_detection_started", self._on_marker_detection_started
        )
        marker_location_controller.add_observer(
            "on_marker_detection_yield", self._on_marker_detection_yield
        )
        marker_location_controller.add_observer(
            "on_marker_detection_ended", self._on_marker_detection_ended
        )

    def create_row(self):
        elements = [self._create_marker_location_bars()]
        if self._marker_location_controller.is_running_detection:
            elements.append(self._create_progress_indication())

        return Row(label=self.timeline_label, elements=elements)

    def _create_marker_location_bars(self):
        bar_positions = [
            marker_location["timestamp"]
            for marker_location in self._marker_locations.result.values()
            if marker_location["markers"]
        ]
        return BarsElementTs(bar_positions, color_rgba=(1.0, 1.0, 1.0, 0.5))

    def _create_progress_indication(self):
        progress = self._marker_location_controller.detection_progress
        return RangeElementFrameIdx(
            from_idx=self._frame_start,
            to_idx=int(self._frame_start + self._frame_count * progress),
            color_rgba=(1.0, 0.5, 0.5, 0.5),
        )

    def _on_marker_detection_started(self):
        self._frame_start, frame_end = self._marker_locations.frame_index_range
        self._frame_count = frame_end - self._frame_start + 1

    def _on_storage_changed(self, *args, **kwargs):
        self.render_parent_timeline()

    def _on_marker_detection_yield(self):
        self.render_parent_timeline()

    def _on_marker_detection_ended(self):
        self.render_parent_timeline()


class CameraLocalizerTimeline:
    timeline_label = "Camera localization"

    def __init__(self, camera_localizer_controller, camera_localizer_storage):
        self.render_parent_timeline = None

        camera_localizer_storage.add_observer(
            "load_from_disk", self._on_storage_changed
        )
        camera_localizer_controller.add_observer(
            "set_range_from_current_trim_marks", self._on_ranges_changed
        )
        camera_localizer_controller.add_observer(
            "save_pose_bisector", self._on_data_changed
        )

        self._camera_localizer = camera_localizer_storage.item

    def create_row(self):
        elements = [self._create_localization_range()]
        rows = Row(label=self.timeline_label, elements=elements)
        return rows

    def _create_localization_range(self):
        from_idx, to_idx = self._camera_localizer.frame_index_range
        color = (
            (0.6, 0.8, 0.4, 0.8)
            if self._camera_localizer.calculated
            else (0.6, 0.8, 0.4, 0.4)
        )
        return RangeElementFrameIdx(from_idx, to_idx, color_rgba=color)

    def _on_ranges_changed(self):
        self.render_parent_timeline()

    def _on_data_changed(self):
        self.render_parent_timeline()

    def _on_storage_changed(self, *args, **kwargs):
        self.render_parent_timeline()
