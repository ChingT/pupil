import background_helper
from marker_tracker_3d import utils
from marker_tracker_3d.optimization.optimization_generator import optimization_generator
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs


class ModelOptimizer:
    def __init__(self, camera_model, update_menu=None):
        self.camera_model = camera_model
        self.origin_marker_id = None

        self.visibility_graphs = VisibilityGraphs(
            self.camera_model, self.origin_marker_id, update_menu
        )

        self.bg_task = None

    def update(self, marker_detections, camera_extrinsics):
        self.visibility_graphs.add_marker_detections(
            marker_detections, camera_extrinsics
        )

        self._run_optimization()

        return self._get_updated_3d_marker_model()

    def _run_optimization(self):
        if not self.bg_task:
            data_for_optimization = self.visibility_graphs.get_data_for_optimization()
            if data_for_optimization:
                args = (self.camera_model, data_for_optimization)
                self.bg_task = background_helper.IPC_Logging_Task_Proxy(
                    name="generator", generator=optimization_generator, args=args
                )

    def _get_updated_3d_marker_model(self):
        optimization_result = self._fetch_optimization_result()
        if optimization_result:
            marker_extrinsics = self.visibility_graphs.get_updated_marker_extrinsics(
                optimization_result
            )
            marker_points_3d = self._get_marker_points_3d(marker_extrinsics)
            return marker_extrinsics, marker_points_3d

    def _fetch_optimization_result(self):
        if self.bg_task:
            for optimization_result in self.bg_task.fetch():
                self.bg_task = None
                return optimization_result

    @staticmethod
    def _get_marker_points_3d(marker_extrinsics):
        if marker_extrinsics is not None:
            marker_points_3d = {
                k: utils.params_to_points_3d(v)[0] for k, v in marker_extrinsics.items()
            }
            return marker_points_3d

    def export_data(self, save_path):
        self.visibility_graphs.vis_graph(save_path)

    def restart(self):
        self.visibility_graphs.reset()
        self.cleanup()

    def cleanup(self):
        if self.bg_task:
            self.bg_task.cancel(timeout=0.001)
            self.bg_task = None
