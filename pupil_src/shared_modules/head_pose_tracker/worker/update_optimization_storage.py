"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


from head_pose_tracker import worker


def run(optimization_storage, bundle_adjustment_result):
    """ process the results of bundle adjustment; update frame_id_to_extrinsics_opt,
    marker_id_to_extrinsics_opt and marker_id_to_points_3d_opt """

    if bundle_adjustment_result:
        update_extrinsics_opt(
            optimization_storage,
            bundle_adjustment_result.frame_id_to_extrinsics,
            bundle_adjustment_result.marker_id_to_extrinsics,
        )
        discard_failed_key_markers(
            optimization_storage, bundle_adjustment_result.frame_ids_failed
        )
    return optimization_storage


def update_extrinsics_opt(
    optimization_storage, frame_id_to_extrinsics, marker_id_to_extrinsics
):
    optimization_storage.frame_id_to_extrinsics_opt.update(frame_id_to_extrinsics)

    for marker_id, extrinsics in marker_id_to_extrinsics.items():
        optimization_storage.marker_id_to_extrinsics_opt[
            marker_id
        ] = extrinsics.tolist()
        optimization_storage.marker_id_to_points_3d_opt[
            marker_id
        ] = worker.utils.convert_marker_extrinsics_to_points_3d(extrinsics).tolist()

    optimization_storage.calculate_centroid()


def discard_failed_key_markers(optimization_storage, frame_ids_failed):
    optimization_storage.all_key_markers = [
        marker
        for marker in optimization_storage.all_key_markers
        if marker.frame_id not in frame_ids_failed
    ]
