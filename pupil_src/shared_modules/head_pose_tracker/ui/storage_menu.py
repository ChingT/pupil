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

from pyglui import ui


class StorageMenu(abc.ABC):
    def __init__(self, storage):
        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

        self._storage = storage
        self.item = self.items[0]

    @property
    @abc.abstractmethod
    def menu_label(self):
        pass

    @abc.abstractmethod
    def _render_custom_ui(self, item, menu):
        pass

    @property
    def items(self):
        # storages are just iterable, but we need things like len() and
        # access by index, so we return a list
        return [item for item in self._storage]

    def render(self):
        self.menu.elements.clear()
        temp_menu = ui.Growing_Menu("Temporary")
        self._render_custom_ui(self.item, temp_menu)
        self.menu.elements.extend(temp_menu.elements)
