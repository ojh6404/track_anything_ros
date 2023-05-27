#!/usr/bin/env python3
import base64

import cv2
import cv_bridge
from dynamic_reconfigure.server import Server
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
from std_msgs.msg import String

# from track_anything_ros.track_anything import TrackAnything

from track_anything_ros.track_anything import TrackAnything
from track_anything_ros.utils import util as TrackAnythingUtil


import requests
import gdown
import os
import sys

import torch
import time
import argparse


class InstanceSegmentationNode(ConnectionBasedTransport):
    def __init__(self, args):
        super(InstanceSegmentationNode, self).__init__()

        self.track_anything = TrackAnything(
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            xmem_checkpoint=args.xmem_checkpoint,
            e2fgvi_checkpoint=args.e2fgvi_checkpoint,
            device=args.device,
        )

        # self.pub_indices = self.advertise(
        #     "~output/cluster_indices", ClusterPointIndices, queue_size=1
        # )
        # self.pub_labels = self.advertise("~output/labels", LabelArray, queue_size=1)
        # self.pub_lbl_cls = self.advertise("~output/label_cls", Image, queue_size=1)
        # self.pub_lbl_ins = self.advertise("~output/label_ins", Image, queue_size=1)
        # self.pub_rects = self.advertise("~output/rects", RectArray, queue_size=1)
        # self.pub_class = self.advertise(
        #     "~output/class", ClassificationResult, queue_size=1
        # )

        # self.pub_segment_img = self.advertise("/test_img", Image, queue_size=1)
        self.pub_segment_img = self.advertise("/test_pub", Image, queue_size=1)

        self.bridge = cv_bridge.CvBridge()

    def subscribe(self):
        # self.sub = rospy.Subscriber(
        #     "~input", Image, self.callback, queue_size=1, buff_size=2**24
        # )
        # TODO
        self.sub = rospy.Subscriber(
            "/kinect_head/rgb/image_raw",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )

    def unsubscribe(self):
        self.sub.unregister()

    def callback(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        out_img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        out_img_msg.header = img_msg.header
        self.pub_segment_img.publish(out_img_msg)

        # data = inference(img, self.score_thresh, url=self.url)
        # target_names = data["class_names"]
        # data = data["response"]
        # labels = np.array(data["labels"], "i")
        # scores = np.array(data["scores"], "f")
        # bboxes = np.array(data["bboxes"], "f")
        # label_names = data["label_names"]
        # b64encoded = data["masks"]
        # bin = base64.b64decode(bin)
        # bin = b64encoded.split(",")[-1]
        # bin = np.frombuffer(bin, np.uint8)
        # masks = bin.reshape(-1, data["height"], data["width"])

        # msg_indices = ClusterPointIndices(header=img_msg.header)
        # msg_labels = LabelArray(header=img_msg.header)

        # R, H, W = masks.shape
        # mask_indices = np.array(np.arange(H * W).reshape(H, W), dtype=np.int32)

        # for ins_id, (mask, label, class_name) in enumerate(
        #     zip(masks, labels, label_names)
        # ):
        #     indices = mask_indices[mask > 0]
        #     indices_msg = PointIndices(header=img_msg.header, indices=indices)
        #     msg_indices.cluster_indices.append(indices_msg)
        #     msg_labels.labels.append(Label(id=label, name=class_name))

        # # -1: label for background
        # if len(masks) > 0:
        #     lbl_cls = np.max((masks > 0) * (labels.reshape(-1, 1, 1) + 1) - 1, axis=0)
        #     lbl_cls = np.array(lbl_cls, dtype=np.int32)
        #     lbl_ins = np.max(
        #         (masks > 0) * (np.arange(R).reshape(-1, 1, 1) + 1) - 1, axis=0
        #     )
        #     lbl_ins = np.array(lbl_ins, dtype=np.int32)
        # else:
        #     lbl_cls = -np.ones(img.shape[:2], dtype=np.int32)
        #     lbl_ins = -np.ones(img.shape[:2], dtype=np.int32)

        # self.pub_indices.publish(msg_indices)
        # self.pub_labels.publish(msg_labels)
        # msg_lbl_cls = bridge.cv2_to_imgmsg(lbl_cls)
        # msg_lbl_ins = bridge.cv2_to_imgmsg(lbl_ins)
        # msg_lbl_cls.header = msg_lbl_ins.header = img_msg.header
        # self.pub_lbl_cls.publish(msg_lbl_cls)
        # self.pub_lbl_ins.publish(msg_lbl_ins)

        # cls_msg = ClassificationResult(
        #     header=img_msg.header,
        #     classifier=self.classifier_name,
        #     target_names=target_names,
        #     labels=labels,
        #     label_names=label_names,
        #     label_proba=scores,
        # )

        # rects_msg = RectArray(header=img_msg.header)
        # for bbox in bboxes:
        #     rect = Rect(
        #         x=int(bbox[0]),
        #         y=int(bbox[1]),
        #         width=int(bbox[2] - bbox[0]),
        #         height=int(bbox[3] - bbox[1]),
        #     )
        #     rects_msg.rects.append(rect)
        # self.pub_rects.publish(rects_msg)
        # self.pub_class.publish(cls_msg)


if __name__ == "__main__":
    # TODO
    model_dir = "/home/leus/ros/catkin_ws/src/jsk_robocup/track_anything_ros/scripts/track_anything_ros/checkpoints/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sam_model_type", type=str, default="vit_b")
    parser.add_argument(
        "--sam_checkpoint", type=str, default=model_dir + "sam_vit_b_01ec64.pth"
    )
    parser.add_argument(
        "--xmem_checkpoint", type=str, default=model_dir + "XMem-s012.pth"
    )
    parser.add_argument(
        "--e2fgvi_checkpoint", type=str, default=model_dir + "E2FGVI-HQ-CVPR22.pth"
    )
    args = parser.parse_args()

    rospy.init_node("instance_segmentation")
    node = InstanceSegmentationNode(args=args)
    rospy.spin()
