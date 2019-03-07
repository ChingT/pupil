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
import multiprocessing
import os
import platform

import tasklib
import video_capture
from head_pose_tracker import worker

if platform.system() == "Darwin":
    mp = multiprocessing.get_context("fork")
else:
    mp = multiprocessing.get_context()

logger = logging.getLogger(__name__ + " with pid: " + str(os.getpid()))


class Global_Container(object):
    pass


class OfflineObservationController:
    def __init__(
        self,
        controller_storage,
        model_storage,
        camera_intrinsics,
        task_manager,
        source_path,
        video_length,
        min_n_markers_per_frame=2,
    ):

        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._min_n_markers_per_frame = min_n_markers_per_frame
        self._source_path = source_path
        self._video_length = video_length

        self.cache_seek_idx = mp.Value("i", 0)
        self.cache_filler = None

        self._init_marker_cache()
        self._init_camera_extrinsics_cache()

        self._decide_key_markers = worker.DecideKeyMarkers(controller_storage)

    def _init_camera_extrinsics_cache(self):
        previous_state = [None for _ in range(self._video_length)]
        self._controller_storage.camera_extrinsics_cache = worker.Cache(previous_state)

    def _init_marker_cache(self):
        previous_marker_cache, version = self._controller_storage.load_marker_cache()

        if previous_marker_cache is None:
            self._recalculate_marker_cache()
        elif version != self._controller_storage.MARKER_CACHE_VERSION:
            logger.debug("Marker cache version missmatch. Rebuilding marker cache.")
            self._recalculate_marker_cache()
        else:
            previous_state = [markers for markers in previous_marker_cache]
            self._recalculate_marker_cache(previous_state)
            logger.info("Restored previous marker cache.")

    def _recalculate_marker_cache(self, previous_state=None):
        if previous_state is None:
            previous_state = [None for _ in range(self._video_length)]

        self._controller_storage.marker_cache = worker.Cache(previous_state)

        self.cache_filler = self._background_video_processor(
            self._source_path,
            worker.detect_markers.marker_detection_callable(),
            list(self._controller_storage.marker_cache),
            self.cache_seek_idx,
        )
        self.cache_filler.add_observer("on_exception", tasklib.raise_exception)
        self.cache_filler.add_observer("on_yield", self._update_caches)
        self.cache_filler.add_observer("on_completed", self._on_completed)
        self.cache_filler.start()

    def _background_video_processor(
        self, video_file_path, callable, visited_list, seek_idx=-1
    ):
        bg_task = self._task_manager.create_background_task(
            name="Background Video Processor",
            routine_or_generator_function=self._video_processing_generator,
            args=(video_file_path, callable, seek_idx, visited_list),
        )
        return bg_task

    @staticmethod
    def _video_processing_generator(video_file_path, callable, seek_idx, visited_list):
        logger.info("Started cacher process for Marker Detector")

        cap = video_capture.File_Source(
            Global_Container(), source_path=video_file_path, timing=None
        )

        visited_list = [x is not None for x in visited_list]

        def next_unvisited_idx(frame_idx):
            """
            Starting from the given index, find the next frame that has not been
            processed yet. If no future frames need processing, check from the start.

            Args:
                frame_idx: Index to start search from.

            Returns: Next index that requires processing.

            """
            try:
                visited = visited_list[frame_idx]
            except IndexError:
                visited = True  # trigger search from the start

            if not visited:
                next_unvisited = frame_idx
            else:
                # find next unvisited site in the future
                try:
                    next_unvisited = visited_list.index(False, frame_idx)
                except ValueError:
                    # any thing in the past?
                    try:
                        next_unvisited = visited_list.index(False, 0, frame_idx)
                    except ValueError:
                        # no unvisited sites left. Done!
                        logger.info("Caching completed.")
                        next_unvisited = None
            return next_unvisited

        def handle_frame(frame_idx):
            if frame_idx != cap.get_frame_index() + 1:
                # we need to seek:
                logger.info("Seeking to Frame {}".format(frame_idx))
                try:
                    cap.seek_to_frame(frame_idx)
                except video_capture.FileSeekError:
                    logger.warning("Could not evaluate frame: {}.".format(frame_idx))
                    visited_list[frame_idx] = True  # this frame is now visited.
                    return []

            try:
                frame = cap.get_frame()
            except video_capture.EndofVideoError:
                logger.warning("Could not evaluate frame: {}.".format(frame_idx))
                visited_list[frame_idx] = True
                return []
            return callable(frame)

        while True:
            last_frame_idx = cap.get_frame_index()
            if seek_idx.value != -1:
                assert seek_idx.value < len(
                    visited_list
                ), "The requested seek index is outside of the predefined cache range!"
                last_frame_idx = seek_idx.value
                seek_idx.value = -1
                logger.info(
                    "User required seek. Marker caching at Frame: {}".format(
                        last_frame_idx
                    )
                )

            next_frame_idx = next_unvisited_idx(last_frame_idx)

            if next_frame_idx is None:
                break
            else:
                res = handle_frame(next_frame_idx)
                visited_list[next_frame_idx] = True
                yield next_frame_idx, res

    def _on_completed(self, _):
        self._controller_storage.export_marker_cache()

    def _update_caches(self, item):
        frame_index, marker_id_to_detections = item

        self._controller_storage.marker_cache.update(
            frame_index, marker_id_to_detections
        )

        self._localize(frame_index, marker_id_to_detections)
        self._pick_key_markers(frame_index, marker_id_to_detections)

    def _localize(self, frame_index, marker_id_to_detections):
        camera_extrinsics = worker.localize_camera.localize(
            self._camera_intrinsics,
            marker_id_to_detections,
            self._model_storage.marker_id_to_extrinsics_opt,
            self._controller_storage.camera_extrinsics,
            self._min_n_markers_per_frame,
        )
        self._controller_storage.update_current_camera_extrinsics(camera_extrinsics)
        self._controller_storage.camera_extrinsics_cache.update(
            frame_index, camera_extrinsics
        )

    def _pick_key_markers(self, frame_index, marker_id_to_detections):
        if self._decide_key_markers.run(marker_id_to_detections):
            self._controller_storage.save_key_markers(
                marker_id_to_detections, frame_index
            )

    def run(self, frame):
        markers = self._controller_storage.marker_cache[frame.index]
        camera_extrinsics = self._controller_storage.camera_extrinsics_cache[
            frame.index
        ]

        self._controller_storage.update_current_marker_id_to_detections(markers)
        self._controller_storage.update_current_camera_pose(camera_extrinsics)

        if markers is None:
            self.cache_seek_idx.value = frame.index
