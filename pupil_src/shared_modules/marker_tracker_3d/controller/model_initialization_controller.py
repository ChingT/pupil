import tasklib

from marker_tracker_3d import worker

from observable import Observable


class ModelInitializationController(Observable):
    def __init__(self, camera_intrinsics, task_manager):
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._bg_task = None

    def run(self, data_for_model_init):
        assert not self._bg_task or not self._bg_task.running

        self._bg_task = self._task_manager.create_background_task(
            name="initial_guess",
            routine_or_generator_function=worker.get_initial_guess.calculate,
            args=(self._camera_intrinsics, data_for_model_init),
        )
        self._bg_task.add_observer("on_completed", self.on_model_init_done)
        self._bg_task.add_observer("on_exception", tasklib.raise_exception)

    def on_model_init_done(self, model_init_result):
        pass

    def reset(self):
        if self._bg_task:
            if self._bg_task.running:
                self._bg_task.kill(grace_period=None)
            self._bg_task = None
