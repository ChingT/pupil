import random

import cv2

from marker_tracker_3d import math, utils


def localize(camera_model, frame_id_to_detections, frame_id_to_extrinsics_prv):
    data_for_triangulation = _prepare_data_for_triangulation(
        camera_model, frame_id_to_detections, frame_id_to_extrinsics_prv
    )
    if not data_for_triangulation:
        return None

    marker_extrinsics = _calculate(data_for_triangulation)
    return marker_extrinsics


def _prepare_data_for_triangulation(
    camera_model, frame_id_to_detections, frame_id_to_extrinsics_prv
):
    # frame_ids_available are the id of the frames which have been known
    # and contain the marker which is going to be estimated.
    frame_ids_available = list(
        set(frame_id_to_extrinsics_prv.keys() & set(frame_id_to_detections.keys()))
    )
    if len(frame_ids_available) < 2:
        return None

    id1, id2 = random.sample(frame_ids_available, 2)

    proj_mat1 = utils.get_extrinsic_matrix(frame_id_to_extrinsics_prv[id1])[:3, :4]
    proj_mat2 = utils.get_extrinsic_matrix(frame_id_to_extrinsics_prv[id2])[:3, :4]

    points1 = frame_id_to_detections[id1]["verts"].reshape((4, 1, 2))
    points2 = frame_id_to_detections[id2]["verts"].reshape((4, 1, 2))
    undistort_points1 = camera_model.undistortPoints(points1)
    undistort_points2 = camera_model.undistortPoints(points2)

    data_for_triangulation = proj_mat1, proj_mat2, undistort_points1, undistort_points2
    return data_for_triangulation


def _calculate(data_for_triangulation):
    marker_points_4d = cv2.triangulatePoints(*data_for_triangulation)
    marker_points_3d = cv2.convertPointsFromHomogeneous(marker_points_4d.T)
    marker_points_3d.shape = 4, 3

    rotation_matrix, translation, error = math.svdt(
        A=utils.get_marker_points_3d_origin(), B=marker_points_3d
    )
    # if error is too large, it means the transformation result is bad
    if error > 0.1:
        return None

    rotation = cv2.Rodrigues(rotation_matrix)[0]
    marker_extrinsics = utils.merge_extrinsics(rotation, translation)
    return marker_extrinsics
