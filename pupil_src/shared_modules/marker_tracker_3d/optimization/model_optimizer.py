import tasklib
from marker_tracker_3d.optimization.model_state import ModelState
from marker_tracker_3d.optimization.optimization_routine import optimization_routine
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs


class ModelOptimizer:
    def __init__(self, task_manager, camera_model, save_path):
        self._task_manager = task_manager
        self._camera_model = camera_model

        self.model_state = ModelState(save_path=save_path)

        self._bg_task = None

        self.visibility_graphs = VisibilityGraphs(
            self.model_state, self._camera_model, predetermined_origin_marker_id=None
        )
        self.visibility_graphs.add_observer(
            "on_ready_for_optimization", self._run_optimization
        )

    def add_observations(self, marker_detections, camera_extrinsics):
        self.visibility_graphs.add_observations(marker_detections, camera_extrinsics)

    def _run_optimization(self):
        assert not self._bg_task or not self._bg_task.running

        self._bg_task = self._task_manager.create_background_task(
            name="optimization_routine",
            routine_or_generator_function=optimization_routine,
            args=(self._camera_model, self.model_state),
        )
        self._bg_task.add_observer(
            "on_completed", self.visibility_graphs.process_optimization_results
        )
        self._bg_task.add_observer("on_exception", tasklib.raise_exception)

    def reset(self):
        if self._bg_task:
            if self._bg_task.running:
                self._bg_task.kill(grace_period=None)
            self._bg_task = None

        self.model_state.reset()
        self.visibility_graphs.reset()
