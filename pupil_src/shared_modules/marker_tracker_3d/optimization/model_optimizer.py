import background_helper
from marker_tracker_3d.optimization.model_optimizer_storage import ModelOptimizerStorage
from marker_tracker_3d.optimization.optimization_generator import optimization_generator
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs


class ModelOptimizer:
    def __init__(self, marker_tracker_3d, camera_model):
        self.marker_tracker_3d = marker_tracker_3d
        self.marker_tracker_3d.add_observer("cleanup", self.cleanup)

        self.camera_model = camera_model

        self.model_optimizer_storage = ModelOptimizerStorage()
        self.origin_marker_id = None
        self.visibility_graphs = VisibilityGraphs(
            self.model_optimizer_storage, self.camera_model, self.origin_marker_id
        )

        self.bg_task = None

    def update(self, marker_detections, camera_extrinsics):
        self.visibility_graphs.add_marker_detections(
            marker_detections, camera_extrinsics
        )

        self._run_optimization()

        marker_extrinsics = self._update_marker_extrinsics()
        return marker_extrinsics

    def _run_optimization(self):
        if not self.bg_task:
            data_for_optimization = self.visibility_graphs.get_data_for_optimization()
            if data_for_optimization:
                args = (self.camera_model, data_for_optimization)
                self.bg_task = background_helper.IPC_Logging_Task_Proxy(
                    name="generator", generator=optimization_generator, args=args
                )

    def _update_marker_extrinsics(self):
        optimization_result = self._fetch_optimization_result()
        marker_extrinsics = None
        if optimization_result:
            marker_extrinsics = self.visibility_graphs.get_updated_marker_extrinsics(
                optimization_result
            )
        return marker_extrinsics

    def _fetch_optimization_result(self):
        if self.bg_task:
            for optimization_result in self.bg_task.fetch():
                self.bg_task = None
                return optimization_result

    def restart(self):
        self.visibility_graphs.reset()
        self.cleanup()

    def cleanup(self):
        if self.bg_task:
            self.bg_task.cancel(timeout=0.001)
            self.bg_task = None
