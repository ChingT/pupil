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


class StorageEditMenu(abc.ABC):
    def __init__(self, storage):
        self.menu = ui.Growing_Menu(self.menu_label)
        self.current_item = None
        # when the current element changes, a few menu elements remain (=the selector
        # and things above) and the rest gets deleted and rendered again (=item
        # specific elements)
        self._number_of_static_menu_elements = 0

        self._storage = storage

    @property
    @abc.abstractmethod
    def menu_label(self):
        pass

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

    def render(self):
        if not self.current_item and len(self.items) > 0:
            self.current_item = self.items[0]
        self.menu.elements.clear()
        if len(self.items) > 0:
            self._render_item_selector_and_current_item()

    def _render_item_selector_and_current_item(self):
        self._number_of_static_menu_elements = len(self.menu.elements)
        # apparently, the 'setter' function is only triggered if the selection
        # changes, but not for the initial selection, so we call it manually
        if self.current_item:
            self._on_change_current_item(self.current_item)

    # TODO: implement this with an attribute observer when the feature is available
    def _on_change_current_item(self, item):
        self.current_item = item
        del self.menu.elements[self._number_of_static_menu_elements :]
        temp_menu = ui.Growing_Menu("Temporary")
        self.render_item(item, temp_menu)
        self.menu.elements.extend(temp_menu.elements)
