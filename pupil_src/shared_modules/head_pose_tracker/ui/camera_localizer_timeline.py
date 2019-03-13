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


# TODO: add an observer to refresh the timeline when the selected optimization of a
#  camera localization changes. This is currently not easily possible,
#  but will be easy with attribute observers


class CameraLocalizerTimeline:
    def __init__(
        self,
        camera_localizer_storage,
        camera_localizer_controller,
        optimization_storage,
        optimization_controller,
    ):
        self.render_parent_timeline = None

        self._camera_localizer_storage = camera_localizer_storage
        self._camera_localizer_controller = camera_localizer_controller
        self._optimization_controller = optimization_controller

        self._camera_localizer_storage.add_observer(
            "add", self._on_localizer_storage_changed
        )
        self._camera_localizer_storage.add_observer(
            "rename", self._on_localizer_storage_changed
        )

        self._camera_localizer_controller.add_observer(
            "set_localization_range_from_current_trim_marks",
            self._on_localizer_ranges_changed,
        )
        self._camera_localizer_controller.add_observer(
            "save_all_enabled_localizers", self._on_save_enabled_localizers
        )
        self._optimization_controller.add_observer(
            "set_optimization_range_from_current_trim_marks",
            self._on_optimization_range_changed,
        )

    def create_rows(self):
        rows = []
        for camera_localizer in self._camera_localizer_storage:
            alpha = 0.9
            elements = [
                self._create_localization_range(camera_localizer, alpha),
                self._create_optimization_range(alpha),
            ]
            rows.append(Row(label=camera_localizer.name, elements=elements))
        return rows

    def _create_localization_range(self, camera_localizer, alpha):
        from_idx, to_idx = camera_localizer.localization_index_range
        # TODO: find some final color scheme
        color = (
            [0.3, 0.5, 0.5, alpha]
            # [136 / 255, 92 / 255, 197 / 255, alpha*1.0]
            if camera_localizer.calculate_complete
            else [0.66 * 0.7, 0.86 * 0.7, 0.46 * 0.7, alpha * 0.8]
        )
        return RangeElementFrameIdx(from_idx, to_idx, color_rgba=color, height=10)

    def _create_optimization_range(self, alpha):
        optimization = (
            self._camera_localizer_controller.get_valid_optimization_or_none()
        )
        color = [0.6, 0.2, 0.8, alpha]
        # color = [217 / 255, 95 / 255, 2 / 255, alpha]
        if (
            optimization is not None
            and self._optimization_controller.is_from_same_recording(optimization)
        ):
            from_idx, to_idx = optimization.frame_index_range
            return RangeElementFrameIdx(
                from_idx, to_idx, color_rgba=color, height=3, offset=-3.5
            )
        else:
            return RangeElementFrameIdx(from_idx=0, to_idx=0)

    def _on_localizer_storage_changed(self, *args, **kwargs):
        self.render_parent_timeline()

    def _on_localizer_ranges_changed(self, _):
        self.render_parent_timeline()

    def _on_save_enabled_localizers(self):
        """Triggered when activate_pose changes and localization tasks are complete"""
        self.render_parent_timeline()

    def _on_optimization_range_changed(self, _):
        self.render_parent_timeline()
