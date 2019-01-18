from marker_tracker_3d import localize_camera, localize_markers
from marker_tracker_3d import utils


def calculate(camera_model, data_for_init):
    """ get marker and camera initial guess for bundle adjustment """

    (
        all_novel_markers,
        camera_extrinsics_prv,
        marker_extrinsics_prv,
        frame_ids,
        marker_ids,
    ) = data_for_init

    camera_extrinsics_init = camera_extrinsics_prv
    marker_extrinsics_init = marker_extrinsics_prv

    # The function _calculate_extrinsics calculates camera extrinsics and marker
    # extrinsics iteratively. It is possible that not all of them can be calculated
    # after one run of _calculate_extrinsics, so we need to run it twice.
    for _ in range(2):
        camera_extrinsics_init = _get_camera_extrinsics_init(
            camera_model,
            all_novel_markers,
            camera_extrinsics_init,
            marker_extrinsics_init,
            frame_ids,
            marker_ids,
        )
        marker_extrinsics_init = _get_marker_extrinsics_init(
            camera_model,
            all_novel_markers,
            camera_extrinsics_init,
            marker_extrinsics_init,
            frame_ids,
            marker_ids,
        )

    return camera_extrinsics_init, marker_extrinsics_init


def _get_camera_extrinsics_init(
    camera_model,
    all_novel_markers,
    camera_extrinsics_init,
    marker_extrinsics_init,
    frame_ids,
    marker_ids,
):
    """ calculate camera extrinsics based on the known marker extrinsics
    """
    marker_points_3d = {
        marker_id: utils.extrinsics_to_marker_points_3d(extrinsics)[0]
        for marker_id, extrinsics in marker_extrinsics_init.items()
    }

    frames_ids_not_computed = set(frame_ids) - set(camera_extrinsics_init.keys())
    for frame_id in frames_ids_not_computed:
        marker_id_to_detections = {
            marker.marker_id: {"verts": marker.verts}
            for marker in all_novel_markers
            if (marker.marker_id in marker_ids and marker.frame_id == frame_id)
        }

        camera_extrinsics = localize_camera.localize(
            camera_model, marker_id_to_detections, marker_points_3d
        )
        if camera_extrinsics is not None:
            camera_extrinsics_init[frame_id] = camera_extrinsics

    return camera_extrinsics_init


def _get_marker_extrinsics_init(
    camera_model,
    all_novel_markers,
    camera_extrinsics_init,
    marker_extrinsics_init,
    frame_ids,
    marker_ids,
):
    """ calculate marker extrinsics based on the known camera extrinsics
    """

    marker_ids_not_computed = set(marker_ids) - set(marker_extrinsics_init.keys())
    for marker_id in marker_ids_not_computed:
        frame_id_to_detections = {
            marker.frame_id: {"verts": marker.verts}
            for marker in all_novel_markers
            if (marker.frame_id in frame_ids and marker.marker_id == marker_id)
        }

        marker_extrinsics = localize_markers.localize(
            camera_model, frame_id_to_detections, camera_extrinsics_init
        )
        if marker_extrinsics is not None:
            marker_extrinsics_init[marker_id] = marker_extrinsics

    return marker_extrinsics_init
