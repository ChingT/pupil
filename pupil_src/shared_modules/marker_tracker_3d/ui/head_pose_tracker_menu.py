import logging

from pyglui import ui as ui

from observable import Observable

logger = logging.getLogger(__name__)


class HeadPoseTrackerMenu(Observable):
    def __init__(self, controller, controller_storage, model_storage, plugin):
        self._controller = controller
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._plugin = plugin

        self._open_3d_window = True
        self.show_markers_opt = True
        self.show_camera_frustum = True
        self.show_camera_trace = True
        # TODO: This is only for debug; should be removed later
        self.show_markers_init = False

        plugin.add_observer("init_ui", self._on_init_ui)
        plugin.add_observer("deinit_ui", self._on_deinit_ui)
        model_storage.add_observer("on_origin_marker_id_set", self._render)

    def _on_init_ui(self):
        self._plugin.add_menu()
        self._plugin.menu.label = "Head Pose Tracker"
        self._render()

    def _on_deinit_ui(self):
        self._plugin.remove_menu()

    def _render(self):
        self._plugin.menu.elements.clear()
        self._plugin.menu.extend(
            [
                self._create_intro_text(),
                self._create_origin_marker_text(),
                self._create_min_marker_perimeter_slider(),
                self._create_open_3d_window_switch(),
                self._create_show_markers_opt_switch(),
                # TODO: debug only; to be removed
                self._create_show_markers_init_switch(),
                self._create_show_camera_frustum_switch(),
                self._create_show_camera_trace_switch(),
                self._create_adding_marker_detections_switch(),
                self._create_reset_button(),
                self._create_load_model_button(),
                self._create_export_model_button(),
                # TODO: debug only; to be removed
                self._create_export_visibility_graph_button(),
                self._create_export_camera_traces_button(),
            ]
        )

    def _create_intro_text(self):
        return ui.Info_Text(
            "This plugin outputs current camera pose in relation to the printed "
            "markers in the scene"
        )

    def _create_origin_marker_text(self):
        if self._model_storage.origin_marker_id is None:
            text = "The coordinate system has not yet been built up"
        else:
            text = (
                "The marker with id {} is defined as the origin of the coordinate "
                "system".format(self._model_storage.origin_marker_id)
            )
            logger.info(text)

        return ui.Info_Text(text)

    def _create_min_marker_perimeter_slider(self):
        return ui.Slider(
            "min_marker_perimeter",
            self._controller_storage,
            step=1,
            min=50,
            max=150,
            label="Perimeter of markers",
        )

    def _create_open_3d_window_switch(self):
        return ui.Switch(
            "_open_3d_window",
            self,
            label="3d visualization window",
            setter=self._switch_3d_window,
        )

    def _create_show_markers_opt_switch(self):
        return ui.Switch("show_markers_opt", self, label="show optimized markers")

    # TODO: debug only; to be removed
    def _create_show_markers_init_switch(self):
        return ui.Switch("show_markers_init", self, label="show init markers (debug)")

    def _create_show_camera_frustum_switch(self):
        return ui.Switch("show_camera_frustum", self, label="show camera frustum")

    def _create_show_camera_trace_switch(self):
        return ui.Switch("show_camera_trace", self, label="show camera trace")

    def _create_adding_marker_detections_switch(self):
        return ui.Switch(
            "adding_observations", self._model_storage, label="Adding new observations"
        )

    def _create_reset_button(self):
        return ui.Button(label="reset", function=self._on_reset_button_click)

    def _create_load_model_button(self):
        return ui.Button(
            outer_label="load",
            label="marker tracker 3d model",
            function=self._on_load_marker_tracker_3d_model_button_click,
        )

    def _create_export_model_button(self):
        return ui.Button(
            outer_label="export",
            label="marker tracker 3d model",
            function=self._on_export_marker_tracker_3d_model_button_click,
        )

    # TODO: debug only; to be removed
    def _create_export_visibility_graph_button(self):
        return ui.Button(
            outer_label="export",
            label="visibility graph (debug)",
            function=self._on_export_visibility_graph_button_click,
        )

    def _create_export_camera_traces_button(self):
        return ui.Button(
            outer_label="export",
            label="camera traces",
            function=self._on_export_camera_traces_button_click,
        )

    def _switch_3d_window(self, open_3d_window):
        self._open_3d_window = open_3d_window
        if self._open_3d_window:
            self.on_open_3d_window()
        else:
            self.on_close_3d_window()

    def on_open_3d_window(self):
        pass

    def on_close_3d_window(self):
        pass

    def _on_reset_button_click(self):
        self._controller.reset()
        self._render()

    def _on_load_marker_tracker_3d_model_button_click(self):
        self._controller.load_marker_tracker_3d_model()

    def _on_export_marker_tracker_3d_model_button_click(self):
        self._controller.export_marker_tracker_3d_model()

    def _on_export_visibility_graph_button_click(self):
        self._controller.export_visibility_graph()

    def _on_export_camera_traces_button_click(self):
        self._controller.export_camera_traces()
