import tasklib
from marker_tracker_3d.optimization.model_optimizer_storage import ModelOptimizerStorage
from marker_tracker_3d.optimization.optimization_routine import optimization_routine
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs


class ModelOptimizer:
    def __init__(self, marker_tracker_3d, camera_model):
        self.marker_tracker_3d = marker_tracker_3d

        self.camera_model = camera_model

        self.model_optimizer_storage = ModelOptimizerStorage()
        self.origin_marker_id = None
        self.visibility_graphs = VisibilityGraphs(
            self.model_optimizer_storage, self.camera_model, self.origin_marker_id
        )

        self.bg_task = None
        self.marker_extrinsics = None

    def update(self, marker_detections, camera_extrinsics):
        self.visibility_graphs.add_marker_detections(
            marker_detections, camera_extrinsics
        )

        self._run_optimization()

        return self.marker_extrinsics

    def _run_optimization(self):
        if not self.bg_task or not self.bg_task.running:
            data_for_optimization = self.visibility_graphs.get_data_for_optimization()
            if data_for_optimization:
                self.bg_task = self.marker_tracker_3d.task_manager.create_background_task(
                    name="optimization_routine",
                    routine_or_generator_function=optimization_routine,
                    args=(self.camera_model, data_for_optimization),
                )
                self.bg_task.add_observer(
                    "on_completed", self._update_marker_extrinsics
                )
                self.bg_task.add_observer("on_exception", tasklib.raise_exception)

    def _update_marker_extrinsics(self, optimization_result):
        if not optimization_result:
            return
        self.marker_extrinsics = self.visibility_graphs.get_updated_marker_extrinsics(
            optimization_result
        )

    def restart(self):
        self.marker_extrinsics = None
        self.visibility_graphs.reset()
        if self.bg_task and self.bg_task.running:
            self.bg_task.kill(grace_period=None)
