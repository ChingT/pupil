"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from pyglui import ui


class OnTopMenu:
    """The part of the menu that's above all other menus (marker locations etc.)"""

    def __init__(self):
        pass

    def render(self, menu):
        menu.extend(
            [
                ui.Info_Text(
                    "This plugin allows you to track camera poses in relation to the "
                    "printed markers in the scene. Before using this plugin, make sure "
                    "you have calibrated the camera intrinsics parameters"
                ),
                ui.Info_Text(
                    "First, marker locations are detected. "
                    "Second, based on the detections, markers 3d model is generated. "
                    "Third, camera localizations is calculated."
                ),
            ]
        )
