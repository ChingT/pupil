"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import file_methods as fm
import video_capture
from apriltag.python import apriltag
from methods import normalize


class Empty(object):
    pass


def offline_detection(
    source_path,
    timestamps,
    frame_index_range,
    frame_index_to_num_markers,
    shared_memory,
):
    batch_size = 30
    frame_start, frame_end = frame_index_range
    frame_indices = sorted(
        set(range(frame_start, frame_end + 1)) - set(frame_index_to_num_markers.keys())
    )
    if not frame_indices:
        return

    frame_count = frame_end - frame_start + 1
    shared_memory.progress = (frame_indices[0] - frame_start + 1) / frame_count
    yield None

    def _detect(image):
        apriltag_detections = apriltag_detector.detect(image)
        if apriltag_detections:
            img_size = image.shape[::-1]
            serialized_dicts = [
                fm.Serialized_Dict(
                    python_dict={
                        "id": detection.tag_id,
                        "verts": detection.corners[::-1].tolist(),
                        "centroid": normalize(detection.center, img_size, flip_y=True),
                        "timestamp": timestamp,
                    }
                )
                for detection in apriltag_detections
            ]
            return len(apriltag_detections), serialized_dicts
        else:
            return 0, [fm.Serialized_Dict(python_dict={})]

    apriltag_detector = apriltag.Detector()
    src = video_capture.File_Source(Empty(), source_path, timing=None)

    queue = []
    for frame_index in frame_indices:
        shared_memory.progress = (frame_index - frame_start + 1) / frame_count
        timestamp = timestamps[frame_index]
        src.seek_to_frame(frame_index)
        frame = src.get_frame()
        num_markers, markers = _detect(frame.gray)
        queue.append((timestamp, markers, frame_index, num_markers))

        if len(queue) >= batch_size:
            data = queue[:batch_size]
            del queue[:batch_size]
            yield data

    yield queue
