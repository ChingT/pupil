"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os

import cv2
import numpy as np

import apriltag
import file_methods as fm
import video_capture
from methods import normalize

apriltag_detector = apriltag.Detector()


class Empty(object):
    pass


def get_markers_data(detection, img_size, timestamp, frame_index):
    return {
        "id": detection.tag_id,
        "verts": detection.corners[::-1].tolist(),
        "centroid": normalize(detection.center, img_size, flip_y=True),
        "timestamp": timestamp,
        "frame_index": frame_index,
    }


def _detect(frame):
    image = frame.gray
    apriltag_detections = apriltag_detector.detect(image)
    img_size = image.shape[::-1]
    return [
        get_markers_data(detection, img_size, frame.timestamp, frame.index)
        for detection in apriltag_detections
        if detection.hamming == 0  # and detection.decision_margin >= 40
    ]


def offline_detection(
    source_path, timestamps, frame_index_to_num_markers, debug, shared_memory
):
    batch_size = 30

    frame_start, frame_end = 0, len(timestamps) - 1
    frame_indices = sorted(
        set(range(frame_start, frame_end + 1)) - set(frame_index_to_num_markers.keys())
    )
    if not frame_indices:
        return

    frame_count = frame_end - frame_start + 1
    shared_memory.progress = (frame_indices[0] - frame_start + 1) / frame_count
    yield None

    src = video_capture.File_Source(
        Empty(),
        timing="external",
        source_path=source_path,
        buffered_decoding=True,
        fill_gaps=True,
    )

    if debug:
        debug_img_folder = os.path.splitext(source_path)[0]
        os.makedirs(debug_img_folder, exist_ok=True)

    queue = []
    for frame_index in frame_indices:
        shared_memory.progress = (frame_index - frame_start + 1) / frame_count
        timestamp = timestamps[frame_index]
        try:
            src.seek_to_frame(frame_index)
        except IndexError:
            continue

        frame = src.get_frame()
        detections = _detect(frame)

        if debug:
            img = frame.bgr.copy()
            verts = [
                np.around(detection["verts"]).astype(np.int32)
                for detection in detections
            ]
            cv2.polylines(img, verts, True, (0, 255, 255), thickness=1)
            cv2.imwrite("{}/{}.jpg".format(debug_img_folder, frame_index), img)

        if detections:
            serialized_dicts = [
                fm.Serialized_Dict(detection) for detection in detections
            ]
            queue.append((timestamp, serialized_dicts, frame_index, len(detections)))
        else:
            queue.append((timestamp, [fm.Serialized_Dict({})], frame_index, 0))

        if len(queue) >= batch_size:
            data = queue[:batch_size]
            del queue[:batch_size]
            yield data

    yield queue


def online_detection(frame):
    return _detect(frame)
