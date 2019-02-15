import numpy as np

from marker_tracker_3d import worker

min_n_markers_per_frame = 1


def localize(
    camera_intrinsics,
    marker_id_to_detections,
    marker_id_to_extrinsics,
    camera_extrinsics_prv=None,
    origin_marker_id=None,
):
    # marker_ids_available are the id of the markers which have been known
    # and are detected in this frame.
    marker_ids_available = list(
        set(marker_id_to_extrinsics.keys() & set(marker_id_to_detections.keys()))
    )
    data_for_solvepnp = _prepare_data_for_solvepnp(
        marker_id_to_detections,
        marker_id_to_extrinsics,
        marker_ids_available,
        origin_marker_id,
    )
    camera_extrinsics = _calculate(
        camera_intrinsics, data_for_solvepnp, camera_extrinsics_prv
    )
    return camera_extrinsics


def _prepare_data_for_solvepnp(
    marker_id_to_detections,
    marker_id_to_extrinsics,
    marker_ids_available,
    origin_marker_id,
):
    if len(marker_ids_available) >= min_n_markers_per_frame:
        markers_points_3d = [
            worker.utils.convert_marker_extrinsics_to_points_3d(
                marker_id_to_extrinsics[i]
            )
            for i in marker_ids_available
        ]
        markers_points_2d = [
            marker_id_to_detections[i]["verts"] for i in marker_ids_available
        ]
    elif origin_marker_id in marker_ids_available:
        markers_points_3d = worker.utils.convert_marker_extrinsics_to_points_3d(
            marker_id_to_extrinsics[origin_marker_id]
        )
        markers_points_2d = marker_id_to_detections[origin_marker_id]["verts"]
    else:
        return None

    markers_points_3d = np.array(markers_points_3d).reshape(-1, 4, 3)
    markers_points_2d = np.array(markers_points_2d).reshape(-1, 4, 2)
    data_for_solvepnp = markers_points_3d, markers_points_2d
    return data_for_solvepnp


def _calculate(camera_intrinsics, data_for_solvepnp, camera_extrinsics_prv):
    if not data_for_solvepnp:
        return None

    markers_points_3d, markers_points_2d = data_for_solvepnp

    retval, rotation, translation = _run_solvepnp(
        camera_intrinsics, markers_points_3d, markers_points_2d, camera_extrinsics_prv
    )
    if _check_solvepnp_output_reasonable(
        retval, rotation, translation, markers_points_3d
    ):
        camera_extrinsics = worker.utils.merge_extrinsics(rotation, translation)
        return camera_extrinsics

    # if _run_solvepnp with camera_extrinsics_prv could not output reasonable result,
    # then do it again without camera_extrinsics_prv
    retval, rotation, translation = _run_solvepnp(
        camera_intrinsics, markers_points_3d, markers_points_2d
    )
    if _check_solvepnp_output_reasonable(
        retval, rotation, translation, markers_points_3d
    ):
        camera_extrinsics = worker.utils.merge_extrinsics(rotation, translation)
        return camera_extrinsics
    else:
        return None


def _run_solvepnp(
    camera_intrinsics, markers_points_3d, markers_points_2d, camera_extrinsics_prv=None
):
    assert len(markers_points_3d) == len(markers_points_2d)
    assert markers_points_3d.shape[1:] == (4, 3)
    assert markers_points_2d.shape[1:] == (4, 2)

    if camera_extrinsics_prv is None:
        retval, rotation, translation = camera_intrinsics.solvePnP(
            markers_points_3d, markers_points_2d
        )
    else:
        rotation_prv, translation_prv = worker.utils.split_extrinsics(
            camera_extrinsics_prv
        )
        retval, rotation, translation = camera_intrinsics.solvePnP(
            markers_points_3d,
            markers_points_2d,
            useExtrinsicGuess=True,
            rvec=rotation_prv.copy(),
            tvec=translation_prv.copy(),
        )
    return retval, rotation, translation


def _check_solvepnp_output_reasonable(retval, rotation, translation, pts_3d_world):
    # solvePnP outputs wrong pose estimations sometimes, so it is necessary to check
    # if the rotation and translation from the output of solvePnP is reasonable.
    if not retval:
        return False

    assert rotation.size == 3 and translation.size == 3

    # if magnitude of translation is too large, it is very possible that the output of
    # solvePnP is wrong.
    if (np.abs(translation) > 2e2).any():
        return False

    # the magnitude of rotation should be less than 2*pi
    if (np.abs(rotation) > np.pi * 2).any():
        return False

    # the depth of the markers in the camera coordinate system should be positive,
    # i.e. all seen markers in the frame should be in front of the camera;
    # if not, that implies the output of solvePnP is wrong.
    pts_3d_camera = worker.utils.to_camera_coordinate(
        pts_3d_world, rotation, translation
    )
    if (pts_3d_camera[:, 2] < 1).any():
        return False

    return True
