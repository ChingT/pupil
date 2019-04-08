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
import tasklib.background
import tasklib.background.patches as bg_patches
import video_capture
from apriltag.python import apriltag
from methods import normalize

g_pool = None  # set by the plugin


def create_task(timestamps, marker_locations):
    assert g_pool, "You forgot to set g_pool by the plugin"
    calculated_timestamps = set(marker_locations.markers_bisector.timestamps)
    args = (
        g_pool.capture.source_path,
        timestamps,
        marker_locations.frame_index_range,
        calculated_timestamps,
    )
    name = "Create Apriltag Detection"
    return tasklib.background.create(
        name,
        _detect_apriltags,
        args=args,
        patches=[bg_patches.IPCLoggingPatch()],
        pass_shared_memory=True,
    )


class Empty(object):
    pass


def _detect_apriltags(
    source_path, timestamps, frame_index_range, calculated_timestamps, shared_memory
):
    def _detect(image):
        apriltag_detections = apriltag_detector.detect(image)
        img_size = image.shape[::-1]
        return [
            fm.Serialized_Dict(
                {
                    "id": detection.tag_id,
                    "verts": detection.corners[::-1].tolist(),
                    "centroid": normalize(detection.center, img_size, flip_y=True),
                    "timestamp": timestamp,
                }
            )
            for detection in apriltag_detections
        ]

    apriltag_detector = apriltag.Detector()
    src = video_capture.File_Source(Empty(), source_path, timing=None)

    frame_start, frame_end = frame_index_range
    frame_count = frame_end - frame_start + 1
    for frame_index in range(frame_start, frame_end + 1):
        shared_memory.progress = (frame_index - frame_start + 1) / frame_count

        timestamp = timestamps[frame_index]
        if timestamp in calculated_timestamps:
            yield timestamp, []
        else:
            src.seek_to_frame(frame_index)
            frame = src.get_frame()
            markers = _detect(frame.gray)
            yield timestamp, markers
