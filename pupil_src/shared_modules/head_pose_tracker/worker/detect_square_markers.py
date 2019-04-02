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


g_pool = None  # set by the plugin


def create_task(marker_location):
    assert g_pool, "You forgot to set g_pool by the plugin"

    args = (g_pool.capture.source_path, marker_location.frame_index_range)
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


def _detect_apriltags(source_path, frame_index_range, shared_memory):
    from apriltag_marker_detector import ApriltagMarkerDetector
    import video_capture

    _detector = ApriltagMarkerDetector()

    src = video_capture.File_Source(Empty(), source_path, timing=None)
    frame_start, frame_end = frame_index_range
    frame_count = frame_end - frame_start + 1
    src.seek_to_frame(frame_start)
    while True:
        try:
            frame = src.get_frame()
        except video_capture.EndofVideoError:
            break
        else:
            if frame.index >= frame_end:
                break
            shared_memory.progress = (frame.index - frame_start + 1) / frame_count

            marker_detection = _detector.detect(frame.gray)
            yield {
                "marker_detection": marker_detection,
                "timestamp": frame.timestamp,
                "frame_index": frame.index,
            }
