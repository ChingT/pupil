import collections

from marker_tracker_3d import localize_camera, localize_markers

InitialGuessResult = collections.namedtuple(
    "InitialGuessResult",
    ["frame_id_to_extrinsics_init", "marker_id_to_extrinsics_init"],
)


def calculate(camera_intrinsics, data_for_model_init):
    """ get marker and camera initial guess for bundle adjustment """

    frame_id_to_extrinsics_init = data_for_model_init.frame_id_to_extrinsics_prv
    marker_id_to_extrinsics_init = data_for_model_init.marker_id_to_extrinsics_prv

    # The function _calculate_extrinsics calculates camera extrinsics and marker
    # extrinsics iteratively. It is possible that not all of them can be calculated
    # after one run of _calculate_extrinsics, so we need to run it twice.
    for _ in range(2):
        frame_id_to_extrinsics_init = _get_frame_id_to_extrinsics_init(
            camera_intrinsics,
            data_for_model_init.all_novel_markers,
            frame_id_to_extrinsics_init,
            marker_id_to_extrinsics_init,
            data_for_model_init.frame_ids_to_be_optimized,
            data_for_model_init.marker_ids_to_be_optimized,
        )
        marker_id_to_extrinsics_init = _get_marker_id_to_extrinsics_init(
            camera_intrinsics,
            data_for_model_init.all_novel_markers,
            frame_id_to_extrinsics_init,
            marker_id_to_extrinsics_init,
            data_for_model_init.frame_ids_to_be_optimized,
            data_for_model_init.marker_ids_to_be_optimized,
        )

    if frame_id_to_extrinsics_init and marker_id_to_extrinsics_init:
        model_init_result = InitialGuessResult(
            frame_id_to_extrinsics_init, marker_id_to_extrinsics_init
        )
        return model_init_result
    else:
        return None


def _get_frame_id_to_extrinsics_init(
    camera_intrinsics,
    all_novel_markers,
    frame_id_to_extrinsics_prv,
    marker_id_to_extrinsics_prv,
    frame_ids,
    marker_ids,
):
    """ calculate camera extrinsics based on the known marker extrinsics """

    frame_id_to_extrinsics_init = frame_id_to_extrinsics_prv
    frame_ids_not_computed = set(frame_ids) - set(frame_id_to_extrinsics_prv.keys())
    for frame_id in frame_ids_not_computed:
        marker_id_to_detections = {
            marker.marker_id: {"verts": marker.verts}
            for marker in all_novel_markers
            if (marker.marker_id in marker_ids and marker.frame_id == frame_id)
        }

        camera_extrinsics = localize_camera.localize(
            camera_intrinsics, marker_id_to_detections, marker_id_to_extrinsics_prv
        )
        if camera_extrinsics is not None:
            frame_id_to_extrinsics_init[frame_id] = camera_extrinsics

    return frame_id_to_extrinsics_init


def _get_marker_id_to_extrinsics_init(
    camera_intrinsics,
    all_novel_markers,
    frame_id_to_extrinsics_prv,
    marker_id_to_extrinsics_prv,
    frame_ids,
    marker_ids,
):
    """ calculate marker extrinsics based on the known camera extrinsics """

    marker_id_to_extrinsics_init = marker_id_to_extrinsics_prv
    marker_ids_not_computed = set(marker_ids) - set(marker_id_to_extrinsics_prv.keys())
    for marker_id in marker_ids_not_computed:
        frame_id_to_detections = {
            marker.frame_id: {"verts": marker.verts}
            for marker in all_novel_markers
            if (marker.frame_id in frame_ids and marker.marker_id == marker_id)
        }

        marker_extrinsics = localize_markers.localize(
            camera_intrinsics, frame_id_to_detections, frame_id_to_extrinsics_prv
        )
        if marker_extrinsics is not None:
            marker_id_to_extrinsics_init[marker_id] = marker_extrinsics

    return marker_id_to_extrinsics_init
