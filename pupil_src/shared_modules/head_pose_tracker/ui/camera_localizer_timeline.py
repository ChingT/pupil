"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from plugin_timeline import Row, RangeElementFrameIdx


class CameraLocalizerTimeline:
    def __init__(self, camera_localizer_storage, camera_localizer_controller):
        self.render_parent_timeline = None

        camera_localizer_storage.add_observer("add", self._on_localizer_storage_changed)
        camera_localizer_controller.add_observer(
            "set_range_from_current_trim_marks", self._on_localizer_ranges_changed
        )
        camera_localizer_controller.add_observer(
            "save_pose_bisector", self._on_localizer_data_changed
        )

        self._camera_localizer = camera_localizer_storage.item

    def create_rows(self):
        rows = []
        if self._camera_localizer is not None:
            alpha = 0.9
            elements = [self._create_localization_range(self._camera_localizer, alpha)]
            rows.append(Row(label=self._camera_localizer.name, elements=elements))
        return rows

    def _create_localization_range(self, camera_localizer, alpha):
        from_idx, to_idx = camera_localizer.localization_index_range
        # TODO: find some final color scheme
        color = (
            [0.3, 0.5, 0.5, alpha]
            if camera_localizer.calculate_complete
            else [0.66 * 0.7, 0.86 * 0.7, 0.46 * 0.7, alpha * 0.8]
        )
        return RangeElementFrameIdx(from_idx, to_idx, color_rgba=color, height=10)

    def _on_localizer_storage_changed(self, *args, **kwargs):
        self.render_parent_timeline()

    def _on_localizer_ranges_changed(self, _):
        self.render_parent_timeline()

    def _on_localizer_data_changed(self, _):
        """Triggered when localization tasks are complete"""
        self.render_parent_timeline()
