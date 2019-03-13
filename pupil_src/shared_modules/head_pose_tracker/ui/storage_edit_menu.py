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

from head_pose_tracker import ui as plugin_ui


class StorageEditMenu(plugin_ui.SelectAndRefreshMenu, abc.ABC):
    """
    A SelectAndRefreshMenu that shows the items in a storage.
    """

    def __init__(self, storage):
        super().__init__()
        self._storage = storage

    @abc.abstractmethod
    def _render_custom_ui(self, item, menu):
        pass

    @abc.abstractmethod
    def _item_label(self, item):
        pass

    @property
    def items(self):
        # storages are just iterable, but we need things like len() and
        # access by index, so we return a list
        return [item for item in self._storage]

    @property
    def item_labels(self):
        return [self._item_label(item) for item in self._storage]

    def render_item(self, item, menu):
        self._render_custom_ui(item, menu)
