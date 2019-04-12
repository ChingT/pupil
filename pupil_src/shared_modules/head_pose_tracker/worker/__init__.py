"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from head_pose_tracker.worker.detection_task import offline_detection, online_detection
from head_pose_tracker.worker.optimization_task import (
    offline_optimization,
    online_optimization,
)
from head_pose_tracker.worker.localization_task import (
    offline_localization,
    online_localization,
)
