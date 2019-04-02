"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import logging
import os
import re

import file_methods as fm

logger = logging.getLogger(__name__)


class StorageItem(abc.ABC):
    @property
    @abc.abstractmethod
    def version(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_tuple(tuple_):
        pass

    @property
    @abc.abstractmethod
    def as_tuple(self):
        pass

    @property
    @abc.abstractmethod
    def calculated(self):
        pass


class Storage(abc.ABC):
    def __init__(self, rec_dir, plugin, get_recording_index_range):
        self._rec_dir = rec_dir
        self._get_recording_index_range = get_recording_index_range

        self.item = None
        self.load_from_disk(self._find_file_path())
        if not self.item:
            self._add_default_item()

        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_cleanup(self):
        self.save_to_disk()

    def save_to_disk(self):
        self._save_data_to_file(self._storage_file_path, self.item.as_tuple)

    def _save_data_to_file(self, filepath, data):
        dict_representation = {"version": self._item_class.version, "data": data}
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fm.save_object(dict_representation, filepath)

    def load_from_disk(self, file_path):
        if not file_path:
            return
        item_tuple = self._load_data_from_file(file_path)
        if item_tuple:
            self.item = self._item_class.from_tuple(item_tuple)

    def _load_data_from_file(self, filepath):
        try:
            dict_representation = fm.load_object(filepath)
        except FileNotFoundError:
            return None
        if dict_representation.get("version", None) != self._item_class.version:
            logger.warning(
                "Data in {} is in old file format. Will not load these!".format(
                    filepath
                )
            )
            return None
        return dict_representation.get("data", None)

    def _find_file_path(self):
        return self._storage_file_path

    def _add_default_item(self):
        self.item = self._create_default_item()

    @abc.abstractmethod
    def _create_default_item(self):
        pass

    @property
    @abc.abstractmethod
    def _item_class(self):
        pass

    @property
    @abc.abstractmethod
    def _storage_folder_path(self):
        pass

    @property
    @abc.abstractmethod
    def _storage_file_name(self):
        pass

    @property
    def _storage_file_path(self):
        return os.path.join(self._storage_folder_path, self._storage_file_name)

    @staticmethod
    def _get_valid_filename(file_name):
        """
        Return the given string converted to a string that can be used for a clean
        filename. Remove leading and trailing spaces; convert other spaces to
        underscores; and remove anything that is not an alphanumeric, dash,
        underscore, or dot.
        E.g.: get_valid_filename("john's portrait in 2004.jpg")
        'johns_portrait_in_2004.jpg'

        Copied from Django:
        https://github.com/django/django/blob/master/django/utils/text.py#L219
        """
        file_name = str(file_name).strip().replace(" ", "_")
        # django uses \w instead of _a-zA-Z0-9 but this leaves characters like ä, Ü, é
        # in the filename, which might be problematic
        return re.sub(r"(?u)[^-_a-zA-Z0-9.]", "", file_name)
