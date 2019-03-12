"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from plugin_timeline import Row, BarsElementTs, RangeElementFramePerc


class MarkerLocationTimeline:
    def __init__(self, marker_location_controller, marker_location_storage):
        self.render_parent_timeline = None

        self._marker_location_controller = marker_location_controller
        self._marker_location_storage = marker_location_storage

        self._marker_location_controller.add_observer(
            "on_detection_started", self._on_start_marker_detection
        )
        self._marker_location_storage.add_observer(
            "add", self._on_marker_storage_changed
        )
        self._marker_location_storage.add_observer(
            "delete", self._on_marker_storage_changed
        )
        self._marker_location_storage.add_observer(
            "delete_all", self._on_marker_storage_changed
        )

    def create_row(self):
        elements = []
        if self._marker_location_controller.is_running_detection:
            elements.append(self._create_progress_indication())
        elements.append(self._create_marker_location_bars())
        return Row(label="Marker detection", elements=elements)

    def _create_progress_indication(self):
        progress = self._marker_location_controller.detection_progress
        return RangeElementFramePerc(
            from_perc=0, to_perc=progress, color_rgba=(1.0, 0.5, 0.5, 0.5)
        )

    def _create_marker_location_bars(self):
        bar_positions = [ref.timestamp for ref in self._marker_location_storage]
        return BarsElementTs(bar_positions, color_rgba=(1.0, 1.0, 1.0, 0.5))

    def _on_start_marker_detection(self, detection_task):
        detection_task.add_observer("update", self._on_marker_detection_update)
        detection_task.add_observer("on_ended", self._on_marker_detection_ended)

    def _on_marker_detection_update(self):
        self.render_parent_timeline()

    def _on_marker_detection_ended(self):
        self.render_parent_timeline()

    def _on_marker_storage_changed(self, *args, **kwargs):
        self.render_parent_timeline()
