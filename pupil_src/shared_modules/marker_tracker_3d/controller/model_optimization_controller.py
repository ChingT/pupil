import tasklib
from marker_tracker_3d import worker
from observable import Observable


class ModelOptimizationController(Observable):
    def __init__(self, model_storage, camera_intrinsics, task_manager):
        self._model_storage = model_storage
        self._task_manager = task_manager
        self._bundle_adjustment = worker.BundleAdjustment(camera_intrinsics)
        self._bg_task = None

    def run(self, model_init_result):
        assert not self._bg_task or not self._bg_task.running

        self._bg_task = self._task_manager.create_background_task(
            name="bundle_adjustment",
            routine_or_generator_function=self._bundle_adjustment.calculate,
            args=(model_init_result,),
        )
        self._bg_task.add_observer("on_completed", self.on_model_opt_done)
        self._bg_task.add_observer("on_exception", tasklib.raise_exception)

    def on_model_opt_done(self, model_opt_results):
        pass

    def reset(self):
        if self._bg_task:
            if self._bg_task.running:
                self._bg_task.kill(grace_period=None)
            self._bg_task = None
