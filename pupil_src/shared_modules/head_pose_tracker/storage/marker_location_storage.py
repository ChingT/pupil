"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os

import file_methods as fm
import player_methods as pm
from observable import Observable


class MarkerLocationStorage(Observable):
    def __init__(self, rec_dir, all_timestamps, plugin):
        self._rec_dir = rec_dir
        self._all_timestamps = all_timestamps.tolist()

        self.markers_bisector = pm.Mutable_Bisector()
        self.frame_index_to_num_markers = {}

        self.load_pldata_from_disk()

        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_cleanup(self):
        self.save_pldata_to_disk()

    @property
    def calculated(self):
        return bool(self.markers_bisector)

    def save_pldata_to_disk(self):
        self._save_to_file()

    def _save_to_file(self):
        directory = self._offline_data_folder_path
        file_name = self._pldata_file_name
        os.makedirs(directory, exist_ok=True)
        all_topics = {
            self._all_timestamps[frame_index]: "{}.{}".format(
                frame_index, self.frame_index_to_num_markers[frame_index]
            )
            for frame_index, num_markers in self.frame_index_to_num_markers.items()
        }
        with fm.PLData_Writer(directory, file_name) as writer:
            for marker_ts, marker in zip(
                self.markers_bisector.timestamps, self.markers_bisector.data
            ):
                writer.append_serialized(
                    timestamp=marker_ts,
                    topic=all_topics[marker_ts],
                    datum_serialized=marker.serialized,
                )

    def load_pldata_from_disk(self):
        self._load_from_file()

    def _load_from_file(self):
        directory = self._offline_data_folder_path
        file_name = self._pldata_file_name
        pldata = fm.load_pldata_file(directory, file_name)
        self.markers_bisector = pm.Mutable_Bisector(pldata.data, pldata.timestamps)
        for topic in set(pldata.topics):
            frame_index, num_markers = topic.split(".")
            self.frame_index_to_num_markers[int(frame_index)] = int(num_markers)

    @property
    def _pldata_file_name(self):
        return "marker_locations"

    @property
    def _offline_data_folder_path(self):
        return os.path.join(self._rec_dir, "offline_data")
