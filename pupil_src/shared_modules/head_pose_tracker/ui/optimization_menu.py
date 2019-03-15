"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

from pyglui import ui

from head_pose_tracker import ui as plugin_ui

logger = logging.getLogger(__name__)


class OptimizationMenu(plugin_ui.StorageEditMenu):
    menu_label = "Optimization"

    def __init__(
        self,
        model_storage,
        optimization_storage,
        optimization_controller,
        index_range_as_str,
    ):
        super().__init__(optimization_storage)
        self._model_storage = model_storage
        self._optimization_storage = optimization_storage
        self._optimization_controller = optimization_controller
        self._index_range_as_str = index_range_as_str

        self.menu.collapsed = False

        optimization_controller.add_observer(
            "on_optimization_computed", self._on_optimization_computed
        )

    def _item_label(self, optimization):
        return optimization.name

    def _render_custom_ui(self, optimization, menu):
        if not self._optimization_controller.is_from_same_recording(optimization):
            self._render_ui_optimization_from_other_recording(optimization, menu)
        else:
            self._render_ui_normally(optimization, menu)

    def _render_ui_normally(self, optimization, menu):
        menu.extend(
            [
                self._create_origin_marker_text(),
                self._create_name_input(optimization),
                self._create_range_selector(optimization),
                self._create_optimize_camera_intrinsics_switch(optimization),
                self._create_calculate_button(optimization),
                self._create_status_display(optimization),
            ]
        )

    def _create_origin_marker_text(self):
        if self._model_storage.origin_marker_id is None:
            text = "The coordinate system has not yet been built up"
        else:
            text = (
                "The marker with id {} is defined as the origin of the coordinate "
                "system".format(self._model_storage.origin_marker_id)
            )
        return ui.Info_Text(text)

    def _create_name_input(self, optimization):
        return ui.Text_Input(
            "name", optimization, label="Name", setter=self._on_name_change
        )

    def _create_range_selector(self, optimization):
        range_string = "Collect Markers in: " + self._index_range_as_str(
            optimization.frame_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set From Trim Marks",
            function=self._on_set_index_range_from_trim_marks,
        )

    def _create_optimize_camera_intrinsics_switch(self, optimization):
        switch = ui.Switch(
            "optimize_camera_intrinsics",
            optimization,
            label="Optimize camera intrinsics",
            setter=self._on_optimize_camera_intrinsics_changed,
        )
        if "%" in optimization.status:
            switch.read_only = True

        return switch

    def _create_status_display(self, optimization):
        return ui.Text_Input("status", optimization, label="Status", setter=lambda _: _)

    def _create_calculate_button(self, optimization):
        return ui.Button(
            label="Recalculate" if optimization.result else "Calculate",
            function=self._on_click_calculate,
        )

    def _render_ui_optimization_from_other_recording(self, optimization, menu):
        menu.append(
            ui.Info_Text(
                self._info_text_for_optimization_from_other_recording(optimization)
            )
        )

    def _info_text_for_optimization_from_other_recording(self, optimization):
        if optimization.result:
            return (
                "This optimization was copied from another recording. "
                "It is ready to be used in camera localizers."
            )
        else:
            return (
                "This optimization was copied from another recording, but you "
                "cannot use it here, because it is not calculated yet. Please go "
                "back to the original recording, calculate the optimization, "
                "and copy it again."
            )

    def _render_ui_online_optimization(self, menu):
        menu.append(ui.Info_Text(self._info_text_for_online_optimization()))

    def _info_text_for_online_optimization(self):
        return (
            "This optimization was created before or during the recording. "
            "It is ready to be used in camera localizers."
        )

    def _on_name_change(self, new_name):
        self._optimization_storage.rename(self.current_item, new_name)
        # we need to render the menu again because otherwise the name in the selector
        # is not refreshed
        self.render()

    def _on_set_index_range_from_trim_marks(self):
        self._optimization_controller.set_optimization_range_from_current_trim_marks(
            self.current_item
        )
        self.render()

    def _on_optimize_camera_intrinsics_changed(self, new_value):
        self.current_item.optimize_camera_intrinsics = new_value

    def _on_click_calculate(self):
        self._optimization_controller.calculate(self.current_item)
        self.render()

    def _on_optimization_computed(self):
        self.render()
