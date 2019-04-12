"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from online_head_pose_tracker import worker


def run(markers_3d_model_storage, bundle_adjustment_result):
    """ process the results of bundle adjustment; update frame_id_to_extrinsics,
    marker_id_to_extrinsics and marker_id_to_points_3d """

    if bundle_adjustment_result:
        update_extrinsics_opt(
            markers_3d_model_storage,
            bundle_adjustment_result.frame_id_to_extrinsics,
            bundle_adjustment_result.marker_id_to_extrinsics,
        )
        discard_failed_key_markers(
            markers_3d_model_storage, bundle_adjustment_result.frame_ids_failed
        )


def update_extrinsics_opt(
    markers_3d_model_storage, frame_id_to_extrinsics_opt, marker_id_to_extrinsics_opt
):
    markers_3d_model_storage.frame_id_to_extrinsics.update(frame_id_to_extrinsics_opt)

    for marker_id, extrinsics in marker_id_to_extrinsics_opt.items():
        markers_3d_model_storage.marker_id_to_extrinsics[
            marker_id
        ] = extrinsics.tolist()
        markers_3d_model_storage.marker_id_to_points_3d[
            marker_id
        ] = worker.utils.convert_marker_extrinsics_to_points_3d(extrinsics).tolist()


def discard_failed_key_markers(markers_3d_model_storage, frame_ids_failed):
    markers_3d_model_storage.all_key_markers = [
        marker
        for marker in markers_3d_model_storage.all_key_markers
        if marker.frame_id not in frame_ids_failed
    ]
