"""
Receive world camera data from Pupil using ZMQ.
Make sure the frame publisher plugin is loaded and confugured to gray or rgb
"""

import numpy as np
import zmq
from msgpack import unpackb, packb

from camera_extrinsics_measurer import camera_names, PI_device
from camera_extrinsics_measurer.live_camera_extrinsics_measurer import (
    Live_Camera_Extrinsics_Measurer,
)
from camera_models import load_intrinsics


# camera_names_PI = ["world", "right", "left"]


class Empty(object):
    pass


# send notification:
def notify(notification):
    """Sends ``notification`` to Pupil Remote"""
    topic = "notify." + notification["subject"]
    payload = packb(notification, use_bin_type=True)
    req.send_string(topic, flags=zmq.SNDMORE)
    req.send(payload)
    return req.recv_string()


def recv_from_sub():
    """Recv a message with topic, payload.
    Topic is a utf-8 encoded string. Returned as unicode object.
    Payload is a msgpack serialized dict. Returned as a python dict.
    Any addional message frames will be added as a list
    in the payload dict with key: '__raw_data__' .
    """
    topic = sub.recv_string()
    payload = unpackb(sub.recv(), encoding="utf-8")
    extra_frames = []
    while sub.get(zmq.RCVMORE):
        extra_frames.append(sub.recv())
    if extra_frames:
        payload["__raw_data__"] = extra_frames
    return topic, payload


context = zmq.Context()
# open a req port to talk to pupil
addr = "127.0.0.1"  # remote ip or localhost
req_port = "50020"  # same as in the pupil remote gui
req = context.socket(zmq.REQ)
req.connect("tcp://{}:{}".format(addr, req_port))
# ask for the sub port
req.send_string("SUB_PORT")
sub_port = req.recv_string()

# Start frame publisher with format BGR
notify(
    {"subject": "start_plugin", "name": "Frame_Publisher", "args": {"format": "gray"}}
)

# open a sub port to listen to pupil
sub = context.socket(zmq.SUB)
sub.connect("tcp://{}:{}".format(addr, sub_port))

# set subscriptions to topics
# recv just pupil/gaze/notifications
sub.setsockopt_string(zmq.SUBSCRIBE, "frame.")

intrinscis_dir = "/home/ch/recordings/five-boards/intrinscis/{}".format(PI_device)
plmodel_dir = "/home/ch/recordings/five-boards/"

intrinsics_dict = {
    "world": load_intrinsics(intrinscis_dir, "world", (1088, 1080)),
    "eye0": load_intrinsics(intrinscis_dir, "eye0", (400, 400)),
    "eye1": load_intrinsics(intrinscis_dir, "eye1", (400, 400)),
}
live_camera_extrinsics_measurer = Live_Camera_Extrinsics_Measurer(
    plmodel_dir, intrinsics_dict
)


def get_frame(msg):
    gray = np.frombuffer(msg["__raw_data__"][0], dtype=np.uint8).reshape(
        msg["height"], msg["width"]
    )
    frame = Empty()
    # frame.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame.gray = gray
    frame.timestamp = msg["timestamp"]
    frame.index = msg["index"]
    return frame


while True:
    topic, message = recv_from_sub()
    camera_name = topic.replace("frame", "").replace(".", "")

    if camera_name in camera_names:
        live_camera_extrinsics_measurer.recent_events(get_frame(message), camera_name)
