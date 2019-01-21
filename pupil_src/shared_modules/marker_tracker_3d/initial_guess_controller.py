import tasklib
from marker_tracker_3d.optimization import initial_guess

from observable import Observable


class InitialGuessController(Observable):
    def __init__(self, model_optimization_storage, camera_model, task_manager):
        self._model_optimization_storage = model_optimization_storage
        self._camera_model = camera_model
        self._task_manager = task_manager

        self._bg_task = None

    def run(self, data_for_init):
        assert not self._bg_task or not self._bg_task.running

        self._bg_task = self._task_manager.create_background_task(
            name="initial_guess",
            routine_or_generator_function=initial_guess.calculate,
            args=(self._camera_model, data_for_init),
        )
        self._bg_task.add_observer("on_completed", self.got_initial_guess)
        self._bg_task.add_observer("on_exception", tasklib.raise_exception)

    def got_initial_guess(self, initial_guess_result):
        frame_id_to_extrinsics_init, marker_id_to_extrinsics_init = initial_guess_result

        data_for_opt = (
            self._model_optimization_storage.all_novel_markers,
            frame_id_to_extrinsics_init,
            marker_id_to_extrinsics_init,
        )
        if frame_id_to_extrinsics_init and marker_id_to_extrinsics_init:
            self.on_got_data_for_opt(data_for_opt)
        else:
            self.on_initial_guess_failed()

    def on_got_data_for_opt(self, data_for_opt):
        pass

    def on_initial_guess_failed(self):
        pass

    def reset(self):
        if self._bg_task:
            if self._bg_task.running:
                self._bg_task.kill(grace_period=None)
            self._bg_task = None
