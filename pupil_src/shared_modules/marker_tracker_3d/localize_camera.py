import numpy as np

from marker_tracker_3d import utils

min_n_markers_per_frame = 1
max_camera_trace_distance = 10


def get(camera_model, data, camera_extrinsics_prv=None):
    if len(data) == 2:
        marker_points_3d, marker_points_2d = _prepare_marker_points_for_loc(*data)
    else:
        marker_points_3d, marker_points_2d = _prepare_marker_points(*data)

    camera_extrinsics = _estimate(
        camera_model, marker_points_3d, marker_points_2d, camera_extrinsics_prv
    )
    return camera_extrinsics


def _estimate(
    camera_model, marker_points_3d, marker_points_2d, camera_extrinsics_prv=None
):
    # calculate camera_extrinsics only when the number of markers is
    # greater than or equal to _min_n_markers_per_frame
    if len(marker_points_3d) < min_n_markers_per_frame:
        return None

    retval, rotation, translation = _run_solvepnp(
        camera_model, marker_points_3d, marker_points_2d, camera_extrinsics_prv
    )

    if _check_solvepnp_output_reasonable(
        retval, rotation, translation, marker_points_3d
    ):
        camera_extrinsics = utils.merge_extrinsics(rotation, translation)
        if _check_camera_trace_reasonable(camera_extrinsics, camera_extrinsics_prv):
            return camera_extrinsics

    return None


def _prepare_marker_points_for_loc(marker_id_to_detections, marker_extrinsics_opt_dict):
    markers_id_available = marker_id_to_detections.keys() & set(
        marker_extrinsics_opt_dict.keys()
    )

    marker_points_3d = utils.extrinsics_to_marker_points_3d(
        [marker_extrinsics_opt_dict[i] for i in markers_id_available]
    )
    marker_points_2d = np.array(
        [marker_id_to_detections[i]["verts"] for i in markers_id_available]
    )

    return marker_points_3d, marker_points_2d


def _prepare_marker_points(
    frame_index,
    frame_indices,
    marker_indices,
    markers_points_2d_detected,
    marker_extrinsics_dict,
):
    markers_index_available = list(
        set(marker_extrinsics_dict.keys())
        & set(marker_indices[frame_indices == frame_index])
    )

    marker_points_3d = np.array(
        [
            utils.extrinsics_to_marker_points_3d(marker_extrinsics_dict[i])[0]
            for i in markers_index_available
        ]
    )

    marker_points_2d = np.array(
        [
            markers_points_2d_detected[
                np.bitwise_and(frame_indices == frame_index, marker_indices == i)
            ][0]
            for i in markers_index_available
        ]
    )
    return marker_points_3d, marker_points_2d


def _run_solvepnp(
    camera_model, marker_points_3d, marker_points_2d, camera_extrinsics_prv
):
    assert marker_points_3d.shape[1:] == (4, 3) and marker_points_2d.shape[1:] == (4, 2)
    assert len(marker_points_3d) == len(marker_points_2d)

    if camera_extrinsics_prv is None:
        retval, rotation, translation = camera_model.solvePnP(
            marker_points_3d, marker_points_2d
        )
    else:
        rotation_prv, translation_prv = utils.split_extrinsics(camera_extrinsics_prv)
        retval, rotation, translation = camera_model.solvePnP(
            marker_points_3d,
            marker_points_2d,
            useExtrinsicGuess=True,
            rvec=rotation_prv,
            tvec=translation_prv,
        )
    return retval, rotation, translation


def _check_solvepnp_output_reasonable(retval, rotation, translation, pts_3d_world):
    if not retval:
        return False

    # solvePnP outputs wrong pose estimations sometimes, so it is necessary to check
    # if the rotation and translation from the output of solvePnP is reasonable.

    assert rotation.size == 3 and translation.size == 3

    # the absolute values of rotation should be less than 2*pi
    if (np.abs(rotation) > np.pi * 2).any():
        return False

    # the depth of the markers in the camera coordinate system should be positive,
    # i.e. all seen markers in the frame should be in front of the camera;
    # if not, that implies the output of solvePnP is wrong.
    pts_3d_camera = utils.to_camera_coordinate(pts_3d_world, rotation, translation)
    if (pts_3d_camera.reshape(-1, 3)[:, 2] < 0).any():
        return False

    return True


def _check_camera_trace_reasonable(camera_extrinsics, camera_extrinsics_prv):
    # If one of camera_extrinsics and camera_extrinsics_prv is None,
    # then just ignore this checking step.
    if camera_extrinsics is None or camera_extrinsics_prv is None:
        return True

    # If the camera position is too far from the previous camera position,
    # then it is very likely the output from solvePnP is wrong estimation.
    camera_trace = utils.get_camera_trace_from_camera_extrinsics(camera_extrinsics)
    camera_trace_prv = utils.get_camera_trace_from_camera_extrinsics(
        camera_extrinsics_prv
    )
    camera_trace_distance = np.linalg.norm(camera_trace - camera_trace_prv)
    if camera_trace_distance > max_camera_trace_distance:
        return False
    else:
        return True
