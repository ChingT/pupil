"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections

from head_pose_tracker import worker

InitialGuessResult = collections.namedtuple(
    "InitialGuessResult",
    [
        "key_markers",
        "frame_id_to_extrinsics",
        "marker_id_to_extrinsics",
        "origin_marker_id",
    ],
)


def calculate(camera_intrinsics, data_for_model_init):
    """ get marker and camera initial guess for bundle adjustment """

    try:
        frame_id_to_extrinsics_init = data_for_model_init.frame_id_to_extrinsics_prv
        marker_id_to_extrinsics_init = data_for_model_init.marker_id_to_extrinsics_prv
    except AttributeError:
        return None

    # The function _calculate_extrinsics calculates camera extrinsics and marker
    # extrinsics iteratively. It is possible that not all of them can be calculated
    # after one run of _calculate_extrinsics, so we need to run it twice.
    for _ in range(2):
        frame_id_to_extrinsics_init = _get_frame_id_to_extrinsics_init(
            camera_intrinsics,
            data_for_model_init.key_markers,
            frame_id_to_extrinsics_init,
            marker_id_to_extrinsics_init,
            data_for_model_init.frame_ids_to_be_optimized,
        )
        marker_id_to_extrinsics_init = _get_marker_id_to_extrinsics_init(
            camera_intrinsics,
            data_for_model_init.key_markers,
            frame_id_to_extrinsics_init,
            marker_id_to_extrinsics_init,
            data_for_model_init.marker_ids_to_be_optimized,
        )

    key_markers = [
        marker
        for marker in data_for_model_init.key_markers
        if (
            marker.frame_id in frame_id_to_extrinsics_init.keys()
            and marker.marker_id in marker_id_to_extrinsics_init.keys()
        )
    ]
    if not key_markers:
        return None

    model_init_result = InitialGuessResult(
        key_markers,
        frame_id_to_extrinsics_init,
        marker_id_to_extrinsics_init,
        data_for_model_init.origin_marker_id,
    )
    return model_init_result


def _get_frame_id_to_extrinsics_init(
    camera_intrinsics,
    key_markers,
    frame_id_to_extrinsics_prv,
    marker_id_to_extrinsics_prv,
    frame_ids,
):
    """ calculate camera extrinsics based on the known marker extrinsics """

    frame_id_to_extrinsics_init = {
        frame_id: extrinsics
        for frame_id, extrinsics in frame_id_to_extrinsics_prv.items()
        if frame_id in frame_ids
    }
    frame_ids_not_computed = set(frame_ids) - set(frame_id_to_extrinsics_init.keys())
    for frame_id in frame_ids_not_computed:
        marker_id_to_detections = {
            marker.marker_id: {"verts": marker.verts}
            for marker in key_markers
            if marker.frame_id == frame_id
            and marker.marker_id in marker_id_to_extrinsics_prv.keys()
        }

        camera_extrinsics = worker.solvepnp.localize(
            camera_intrinsics, marker_id_to_detections, marker_id_to_extrinsics_prv
        )
        if camera_extrinsics is not None:
            frame_id_to_extrinsics_init[frame_id] = camera_extrinsics

    return frame_id_to_extrinsics_init


def _get_marker_id_to_extrinsics_init(
    camera_intrinsics,
    key_markers,
    frame_id_to_extrinsics_prv,
    marker_id_to_extrinsics_prv,
    marker_ids,
):
    """ calculate marker extrinsics based on the known camera extrinsics """

    marker_id_to_extrinsics_init = {
        marker_id: extrinsics
        for marker_id, extrinsics in marker_id_to_extrinsics_prv.items()
        if marker_id in marker_ids
    }
    marker_ids_not_computed = set(marker_ids) - set(marker_id_to_extrinsics_init.keys())
    for marker_id in marker_ids_not_computed:
        frame_id_to_detections = {
            marker.frame_id: {"verts": marker.verts}
            for marker in key_markers
            if marker.marker_id == marker_id
            and marker.frame_id in frame_id_to_extrinsics_prv.keys()
        }

        marker_extrinsics = worker.localize_markers.localize(
            camera_intrinsics, frame_id_to_detections, frame_id_to_extrinsics_prv
        )
        if marker_extrinsics is not None:
            marker_id_to_extrinsics_init[marker_id] = marker_extrinsics

    return marker_id_to_extrinsics_init
