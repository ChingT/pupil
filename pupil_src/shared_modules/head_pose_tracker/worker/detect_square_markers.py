"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import tasklib.background
import tasklib.background.patches as bg_patches
from head_pose_tracker import model

g_pool = None  # set by the plugin


def create_task():
    args = (g_pool.capture.source_path,)
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


def _detect_apriltags(source_path, shared_memory):
    from apriltag_marker_detector import ApriltagMarkerDetector
    import video_capture

    _detector = ApriltagMarkerDetector()
    src = video_capture.File_Source(Empty(), source_path, timing=None)
    frame_count = src.get_frame_count()

    while True:
        try:
            frame = src.get_frame()
        except video_capture.EndofVideoError:
            break

        marker_detection = _detector.detect(frame.gray)
        shared_memory.progress = (frame.index + 1) / frame_count

        if marker_detection:
            yield model.MarkerLocation(marker_detection, frame.index, frame.timestamp)
