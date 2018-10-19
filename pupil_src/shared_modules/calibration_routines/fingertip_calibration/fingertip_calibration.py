'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''


import torch
import os,sys
import cv2
import numpy as np
import logging
logger = logging.getLogger(__name__)

from pyglui import ui
from pyglui.cygl import utils as cygl_utils

import audio
from calibration_routines import calibration_plugin_base, finish_calibration
from calibration_routines.fingertip_calibration.models import ssd_lite, unet

if getattr(sys, 'frozen', False):
    weights_root = os.path.join(sys._MEIPASS, "weights")
else:
    weights_root = os.path.join(os.path.split(__file__)[0], "weights")


class Fingertip_Calibration(calibration_plugin_base.Calibration_Plugin):
    """Calibrate gaze parameters using your fingertip.
       Move your head for example horizontally and vertically while gazing at your fingertip
       to quickly sample a wide range gaze angles.
    """
    def __init__(self, g_pool,visualize=True):
        super().__init__(g_pool)
        self.menu = None

        # Initialize CNN pipeline
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Hand Detector cfg
        hand_detector_cfg = {
            'input_size': 225,
            'confidence_thresh': 0.9,
            'max_num_detection': 1,
            'nms_thresh': 0.45,
        }
        # Fingertip Detector cfg
        fingertip_detector_cfg = {
            'confidence_thresh': 0.6,
        }
        self.hand_fingertip_detector = HandFingertipDetector(hand_detector_cfg, fingertip_detector_cfg)
        weights_path = os.path.join(weights_root, "hand_fingertip_detector_model.pkl")
        self.hand_fingertip_detector.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        self.hand_fingertip_detector.eval().to(self.device)
        self.transform = BaseTransform(self.device, size=hand_detector_cfg['input_size'],
                                       mean=(117.77, 115.42, 107.29), std=(72.03, 69.83, 71.43))

        self.collect_tips = False
        self.visualize = visualize
        self.hand_viz = []
        self.finger_viz = []

    def get_init_dict(self):
        return {"visualize": self.visualize}

    def init_ui(self):
        super().init_ui()
        self.menu.label = 'Fingertip Calibration'
        self.menu.append(ui.Info_Text('Calibrate gaze parameters using your fingertip!'))
        self.menu.append(ui.Info_Text('Hold your index finger still at the center of the field of view of the world camera. '
                                      'Move your head horizontally and then vertically while gazing at your fingertip.'
                                      'Then show five fingers to finish the calibration.'))
        if self.device == torch.device("cpu"):
            self.menu.append(ui.Info_Text('* No GPU utilized for fingertip detection network. '
                                          'Note that the frame rate will drop during fingertip detection.'))
        else:
            self.menu.append(ui.Info_Text('* GPUs utilized for fingertip detection network'))

        self.vis_toggle = ui.Thumb('visualize', self, label='V', hotkey='v')
        self.g_pool.quickbar.append(self.vis_toggle)

    def start(self):
        if not self.g_pool.capture.online:
            logger.error("This calibration requires world capture video input.")
            return
        super().start()
        audio.say("Starting Fingertip Calibration")
        logger.info("Starting Fingertip Calibration")

        self.active = True
        self.ref_list = []
        self.pupil_list = []

    def stop(self):
        # TODO: redundancy between all gaze mappers -> might be moved to parent class
        audio.say("Stopping Fingertip Calibration")
        logger.info('Stopping Fingertip Calibration')
        self.active = False
        self.button.status_text = ''
        if self.mode == 'calibration':
            finish_calibration.finish_calibration(self.g_pool, self.pupil_list, self.ref_list)
        elif self.mode == 'accuracy_test':
            self.finish_accuracy_test(self.pupil_list, self.ref_list)
        super().stop()

    def recent_events(self, events):
        frame = events.get('frame')
        if (self.visualize or self.active) and frame:
            img_width, img_height = frame.width, frame.height
            img = frame.img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)

            # Hand Detection and Fingertip detection
            hands, fingertips = self.hand_fingertip_detector(img, img_height, img_width)
            self.hand_viz = []
            self.finger_viz = []
            for hand, fingertip in zip(hands, fingertips):
                detected_fingertips = [p for p in fingertip if p is not None]

                if len(detected_fingertips) > 0:
                    # Hand detections without fingertips are false positives,
                    # so only the hands with detected fingers are visualized
                    self.hand_viz.append(hand)
                    self.finger_viz.append(fingertip)

                    if len(detected_fingertips) == 1 and self.active:
                        x, y = detected_fingertips[0]
                        ref = {
                            'screen_pos': (x, y),
                            'norm_pos': (x / img_width, 1 - (y / img_height)),
                            'timestamp': frame.timestamp,
                        }
                        self.ref_list.append(ref)
                    elif len(detected_fingertips) == 5 and self.active:
                        if self.collect_tips and len(self.ref_list) > 5:
                            self.collect_tips = False
                            self.stop()
                        elif not self.collect_tips:
                            self.collect_tips = True

            if self.active:
                # always save pupil positions
                self.pupil_list.extend(events['pupil'])

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """
        if self.active or self.visualize:
            # Draw hand detection results
            for (x1, y1, x2, y2), fingertips in zip(self.hand_viz, self.finger_viz):
                pts = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]], np.int32)
                cygl_utils.draw_polyline(pts, thickness=3 * self.g_pool.gui_user_scale, color=cygl_utils.RGBA(0., 1., 0., 1.))
                for tip in fingertips:
                    if tip is not None:
                        x, y = tip
                        cygl_utils.draw_progress((x, y), 0., 1.,
                                      inner_radius=25 * self.g_pool.gui_user_scale,
                                      outer_radius=35 * self.g_pool.gui_user_scale,
                                      color=cygl_utils.RGBA(1., 1., 1., 1.),
                                      sharpness=0.9)

                        cygl_utils.draw_points([(x, y)], size=10 * self.g_pool.gui_user_scale,
                                    color=cygl_utils.RGBA(1., 1., 1., 1.),
                                    sharpness=0.9)

    def deinit_ui(self):
        """gets called when the plugin get terminated.
           either voluntarily or forced.
        """
        if self.active:
            self.stop()
        self.g_pool.quickbar.remove(self.vis_toggle)
        self.vis_toggle = None
        super().deinit_ui()


class BaseTransform(object):
    def __init__(self, device, size, mean=None, std=None):
        self.device = device
        self.size = size
        self.resize = torch.nn.Upsample(size=(self.size, self.size), mode='bilinear', align_corners=False)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device) if mean is not None else None
        self.std = torch.tensor(std, dtype=torch.float32, device=device) if std is not None else None

    def __call__(self, image):
        if self.device == torch.device('cpu'):
            image = cv2.resize(image, dsize=(self.size, self.size))
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            if self.mean is not None:
                image -= self.mean
            if self.std is not None:
                image /= self.std
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
        else:
            image = torch.from_numpy(image)
            image = image.to(self.device)
            image = image.float()
            if self.mean is not None:
                image -= self.mean
            if self.std is not None:
                image /= self.std
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
            image = self.resize(image)
        return image


class HandFingertipDetector(torch.nn.Module):
    def __init__(self, hand_detector_cfg, fingertip_detector_cfg):
        super().__init__()
        self.hand_detector = ssd_lite.build_ssd_lite(hand_detector_cfg)
        self.fingertip_detector = unet.UNet(num_classes=10, in_channels=3, depth=4, start_filts=32, up_mode='upsample')
        self.resample = torch.nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        self.fingertip_conf_thresh = fingertip_detector_cfg['confidence_thresh']

    def forward(self, image, img_height, img_width):
        resize_ratio = torch.tensor([image.size(-1)/img_width, image.size(-2)/img_height], dtype=torch.float32, device=image.device)
        hands = []
        fingertips = []

        # Hand Detection
        hand_detections = self.hand_detector(image)[0][1].detach()
        for hand_detection in hand_detections:
            confidence, x1, y1, x2, y2 = hand_detection
            if confidence == 0:
                break

            x1 *= img_width
            x2 *= img_width
            y1 *= img_height
            y2 *= img_height

            tl = torch.tensor([x1, y1], device=image.device)
            br = torch.tensor([x2, y2], device=image.device)
            W, H = br - tl
            crop_len = torch.clamp(max(W, H) * 1.25, 1, min(img_width, img_height))
            crop_center = (br + tl) / 2
            crop_center[0] = torch.clamp(crop_center[0], crop_len / 2, img_width - crop_len / 2)
            crop_center[1] = torch.clamp(crop_center[1], crop_len / 2, img_height - crop_len / 2)
            crop_tl_ori = (crop_center - crop_len / 2)
            crop_br_ori = (crop_tl_ori + crop_len)

            # Fingertip detection
            crop_tl_225 = (crop_tl_ori * resize_ratio).int()
            crop_br_225 = (crop_br_ori * resize_ratio).int()

            image_cropped = image[:, :, crop_tl_225[1]:crop_br_225[1], crop_tl_225[0]:crop_br_225[0]]
            image_cropped = self.resample(image_cropped)

            fingertip_detections = self.fingertip_detector(image_cropped)[0][:5].detach()

            crop_tl_ori = crop_tl_225.type_as(image) / resize_ratio
            crop_br_ori = crop_br_225.type_as(image) / resize_ratio

            fingertip = []
            for i, fingertip_detection in enumerate(fingertip_detections):
                max_index = fingertip_detection.argmax()
                p = max_index / fingertip_detection.shape[1], max_index % fingertip_detection.shape[1]
                if fingertip_detection[p] >= self.fingertip_conf_thresh:
                    py = p[0].type_as(image)
                    px = p[1].type_as(image)
                    ref_y = py / image_cropped.size(-2) * (crop_br_ori[1] - crop_tl_ori[1]) + crop_tl_ori[1]
                    ref_x = px / image_cropped.size(-1) * (crop_br_ori[0] - crop_tl_ori[0]) + crop_tl_ori[0]
                    ref = ref_x.cpu().numpy(), ref_y.cpu().numpy()
                    fingertip.append(ref)
                else:
                    fingertip.append(None)
            hands.append((x1.cpu().numpy(), y1.cpu().numpy(), x2.cpu().numpy(), y2.cpu().numpy()))
            fingertips.append(fingertip)

        return hands, fingertips
