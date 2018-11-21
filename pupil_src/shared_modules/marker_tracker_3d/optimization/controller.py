import logging
import multiprocessing as mp
import os

import numpy as np

import background_helper
from marker_tracker_3d import utils
from marker_tracker_3d.optimization.optimization_generator import optimization_generator

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, storage, on_first_yield=None):
        self.storage = storage
        self.on_first_yield = on_first_yield
        self.first_yield_done = False
        self.frame_count = 0
        self.send_data_interval = 6

        recv_pipe, self.send_pipe = mp.Pipe(False)
        generator_args = (recv_pipe,)
        self.bg_task = background_helper.IPC_Logging_Task_Proxy(
            name="generator", generator=optimization_generator, args=generator_args
        )
        self.send_pipe.send(
            ("basic_models", (self.storage.camera_model, self.storage.marker_model))
        )

    def fetch_extrinsics(self):
        for marker_extrinsics in self.bg_task.fetch():
            if not self.first_yield_done:
                self.on_first_yield()
                self.first_yield_done = True

            self.storage.marker_extrinsics = marker_extrinsics
            self.storage.marker_points_3d = {
                k: self.storage.marker_model.params_to_points_3d(v)[0]
                for k, v in self.storage.marker_extrinsics.items()
            }

            logger.info(
                "{} markers have been registered and updated".format(
                    len(marker_extrinsics)
                )
            )

    def send_marker_data(self):
        self.frame_count += 1
        if self.frame_count > self.send_data_interval:
            self.frame_count = 0
            if self.storage.register_new_markers:
                self.send_pipe.send(
                    ("frame", (self.storage.markers, self.storage.camera_extrinsics))
                )

    def save_data(self):
        if not os.path.exists(self.storage.save_path):
            os.makedirs(self.storage.save_path)

        dist = [
            np.linalg.norm(
                self.storage.camera_trace[i + 1] - self.storage.camera_trace[i]
            )
            if self.storage.camera_trace[i + 1] is not None
            and self.storage.camera_trace[i] is not None
            else np.nan
            for i in range(len(self.storage.camera_trace) - 1)
        ]

        dicts = {"dist": dist, "reprojection_errors": self.storage.reprojection_errors}
        utils.save_params_dicts(save_path=self.storage.save_path, dicts=dicts)

        self.send_pipe.send(("save", self.storage.save_path))
        logger.info("save_data at {}".format(self.storage.save_path))

    def restart(self):
        self.first_yield_done = False
        self.frame_count = 0

        logger.warning("Restart!")
        self.send_pipe.send(("restart", None))

    def cleanup(self):
        if self.bg_task:
            self.bg_task.cancel()
            self.bg_task = None
