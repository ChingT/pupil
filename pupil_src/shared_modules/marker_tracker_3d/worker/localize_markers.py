import itertools

import cv2
import numpy as np

from marker_tracker_3d import worker


def localize(camera_intrinsics, frame_id_to_detections, frame_id_to_extrinsics):
    # frame_ids_available are the id of the frames which have been known
    # and contain the marker which is going to be estimated.
    frame_ids_available = list(
        set(frame_id_to_extrinsics.keys() & set(frame_id_to_detections.keys()))
    )
    for id1, id2 in itertools.combinations(frame_ids_available, 2):
        data_for_triangulation = _prepare_data_for_triangulation(
            camera_intrinsics, frame_id_to_detections, frame_id_to_extrinsics, id1, id2
        )
        marker_extrinsics = _calculate(data_for_triangulation)
        if marker_extrinsics is not None:
            return marker_extrinsics

    return None


def _prepare_data_for_triangulation(
    camera_intrinsics, frame_id_to_detections, frame_id_to_extrinsics, id1, id2
):
    proj_mat1 = worker.utils.get_extrinsic_matrix(frame_id_to_extrinsics[id1])[:3, :4]
    proj_mat2 = worker.utils.get_extrinsic_matrix(frame_id_to_extrinsics[id2])[:3, :4]

    points1 = frame_id_to_detections[id1]["verts"].reshape((4, 1, 2))
    points2 = frame_id_to_detections[id2]["verts"].reshape((4, 1, 2))
    undistort_points1 = camera_intrinsics.undistortPoints(points1)
    undistort_points2 = camera_intrinsics.undistortPoints(points2)

    data_for_triangulation = proj_mat1, proj_mat2, undistort_points1, undistort_points2
    return data_for_triangulation


def _calculate(data_for_triangulation):
    marker_points_3d = _run_triangulation(data_for_triangulation)

    rotation_matrix, translation, error = worker.svdt(
        A=worker.utils.get_marker_points_3d_origin(), B=marker_points_3d
    )

    if _check_triangulate_output_reasonable(translation, error):
        rotation = cv2.Rodrigues(rotation_matrix)[0]
        marker_extrinsics = worker.utils.merge_extrinsics(rotation, translation)
        return marker_extrinsics
    else:
        return None


def _run_triangulation(data_for_triangulation):
    marker_points_4d = cv2.triangulatePoints(*data_for_triangulation)
    marker_points_3d = cv2.convertPointsFromHomogeneous(marker_points_4d.T)
    marker_points_3d.shape = 4, 3
    return marker_points_3d


def _check_triangulate_output_reasonable(translation, error):
    # Sometimes the frames may contain bad marker detection, which could lead to bad
    # triangulation. So it is necessary to check if the output of triangulation is
    # reasonable.

    # if svdt error is too large, it is very possible that the
    # triangulate result is wrong.
    if error > 5e-2:
        return False

    # if magnitude of translation is too large, it is very possible that the
    # triangulate result is wrong.
    if (np.abs(translation) > 2e2).any():
        return False

    return True
