#!/usr/bin/env python3

from __future__ import print_function

import base64

import cv2
import cv_bridge
from dynamic_reconfigure.server import Server
from jsk_perception.cfg import MaskRCNNInstanceSegmentationConfig as Config
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import ClusterPointIndices
from jsk_recognition_msgs.msg import Label
from jsk_recognition_msgs.msg import LabelArray
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from jsk_topic_tools import ConnectionBasedTransport
import numpy as np
from pcl_msgs.msg import PointIndices
import requests
import rospy
from sensor_msgs.msg import Image


try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
except Exception:
    jpeg = None


def encode_image_turbojpeg(img):
    bin = jpeg.encode(img)
    b64encoded = base64.b64encode(bin).decode('ascii')
    return b64encoded


def encode_image_cv2(img, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    b64encoded = base64.b64encode(encimg).decode('ascii')
    return b64encoded


def encode_image(img):
    if jpeg is not None:
        img = encode_image_turbojpeg(img)
    else:
        img = encode_image_cv2(img)
    return img


def inference(
        img,
        score_thresh: float = 0.2,
        url: str = 'http://localhost:8051/detect/'):
    response = requests.post(
        url,
        json={'image':
              {'data': encode_image(img)},
              'score_thresh': float(score_thresh),
        })
    data = response.json()
    return data


class InstanceSegmentationNode(ConnectionBasedTransport):

    def __init__(self):
        super(InstanceSegmentationNode, self).__init__()

        self.url = rospy.get_param(
            '~url',
            'http://localhost:8051/detect/')

        self.classifier_name = rospy.get_param(
            "~classifier_name", rospy.get_name())

        self.score_thresh = rospy.get_param(
            '~score_thresh', Config.defaults['score_thresh'])

        self.srv = Server(Config, self.config_callback)

        self.pub_indices = self.advertise(
            '~output/cluster_indices', ClusterPointIndices, queue_size=1)
        self.pub_labels = self.advertise(
            '~output/labels', LabelArray, queue_size=1)
        self.pub_lbl_cls = self.advertise(
            '~output/label_cls', Image, queue_size=1)
        self.pub_lbl_ins = self.advertise(
            '~output/label_ins', Image, queue_size=1)
        self.pub_rects = self.advertise(
            "~output/rects", RectArray,
            queue_size=1)
        self.pub_class = self.advertise(
            "~output/class", ClassificationResult,
            queue_size=1)

    def subscribe(self):
        self.sub = rospy.Subscriber('~input', Image, self.callback,
                                    queue_size=1, buff_size=2**24)

    def unsubscribe(self):
        self.sub.unregister()

    def config_callback(self, config, level):
        self.score_thresh = config.score_thresh
        return config

    def callback(self, img_msg):
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        data = inference(img, self.score_thresh, url=self.url)
        target_names = data['class_names']
        data = data['response']
        labels = np.array(data['labels'], 'i')
        scores = np.array(data['scores'], 'f')
        bboxes = np.array(data['bboxes'], 'f')
        label_names = data['label_names']
        b64encoded = data['masks']
        bin = b64encoded.split(",")[-1]
        bin = base64.b64decode(bin)
        bin = np.frombuffer(bin, np.uint8)
        masks = bin.reshape(-1, data['height'], data['width'])

        msg_indices = ClusterPointIndices(header=img_msg.header)
        msg_labels = LabelArray(header=img_msg.header)

        R, H, W = masks.shape
        mask_indices = np.array(
            np.arange(H * W).reshape(H, W), dtype=np.int32)

        for ins_id, (mask, label, class_name) in enumerate(
                zip(masks, labels, label_names)):
            indices = mask_indices[mask > 0]
            indices_msg = PointIndices(header=img_msg.header, indices=indices)
            msg_indices.cluster_indices.append(indices_msg)
            msg_labels.labels.append(Label(id=label, name=class_name))

        # -1: label for background
        if len(masks) > 0:
            lbl_cls = np.max(
                (masks > 0)
                * (labels.reshape(-1, 1, 1) + 1) - 1, axis=0)
            lbl_cls = np.array(lbl_cls, dtype=np.int32)
            lbl_ins = np.max(
                (masks > 0) * (np.arange(R).reshape(-1, 1, 1) + 1) - 1,
                axis=0)
            lbl_ins = np.array(lbl_ins, dtype=np.int32)
        else:
            lbl_cls = - np.ones(img.shape[:2], dtype=np.int32)
            lbl_ins = - np.ones(img.shape[:2], dtype=np.int32)

        self.pub_indices.publish(msg_indices)
        self.pub_labels.publish(msg_labels)
        msg_lbl_cls = bridge.cv2_to_imgmsg(lbl_cls)
        msg_lbl_ins = bridge.cv2_to_imgmsg(lbl_ins)
        msg_lbl_cls.header = msg_lbl_ins.header = img_msg.header
        self.pub_lbl_cls.publish(msg_lbl_cls)
        self.pub_lbl_ins.publish(msg_lbl_ins)

        cls_msg = ClassificationResult(
            header=img_msg.header,
            classifier=self.classifier_name,
            target_names=target_names,
            labels=labels,
            label_names=label_names,
            label_proba=scores,
        )

        rects_msg = RectArray(header=img_msg.header)
        for bbox in bboxes:
            rect = Rect(x=int(bbox[0]), y=int(bbox[1]),
                        width=int(bbox[2] - bbox[0]),
                        height=int(bbox[3] - bbox[1]))
            rects_msg.rects.append(rect)
        self.pub_rects.publish(rects_msg)
        self.pub_class.publish(cls_msg)


if __name__ == '__main__':
    rospy.init_node('instance_segmentation')
    node = InstanceSegmentationNode()
    rospy.spin()
