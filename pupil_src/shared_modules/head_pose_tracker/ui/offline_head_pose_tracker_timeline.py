"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


class OfflineHeadPoseTrackerTimeline:
    def __init__(
        self,
        plugin_timeline,
        marker_location_timeline,
        camera_localizer_timeline,
        plugin,
    ):
        self._plugin_timeline = plugin_timeline
        self._marker_location_timeline = marker_location_timeline
        self._camera_localizer_timeline = camera_localizer_timeline

        marker_location_timeline.render_parent_timeline = self.render
        camera_localizer_timeline.render_parent_timeline = self.render

        plugin.add_observer("init_ui", self._on_init_ui)

    def render(self):
        self._plugin_timeline.clear_rows()
        self._plugin_timeline.add_row(self._marker_location_timeline.create_row())
        for row in self._camera_localizer_timeline.create_rows():
            self._plugin_timeline.add_row(row)
        self._plugin_timeline.refresh()

    def _on_init_ui(self):
        self.render()
