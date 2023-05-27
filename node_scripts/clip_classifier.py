#!/usr/bin/env python3

import cv_bridge
from jsk_topic_tools import ConnectionBasedTransport
import numpy as np
import rospy
from sensor_msgs.msg import Image
import message_filters
import clip
import cv2
import torch
import torch.nn as nn
from PIL import Image as PImage
import numpy as np

from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from jsk_recognition_msgs.msg import ClassificationResult


class Linear(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.softmax(x, 1)
        return x


def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = PImage.fromarray(new_image)
    return new_image

def boxes_to_rectsarray(boxes, header=None):
    rects_msg = RectArray(header=header)
    rects_msg.rects = [
        Rect(x=x, y=y, width=w, height=h) for x, y, w, h in boxes
    ]
    return rects_msg


class ClipClassifierNode(ConnectionBasedTransport):

    def __init__(self):
        super(ClipClassifierNode, self).__init__()

        device = rospy.get_param('~device', '-1')
        if device < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        rospy.loginfo("loading clip model")
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        rospy.loginfo("loading clip model end")
        rospy.loginfo("loading pytorch linear model")
        self.linear_model= torch.load(rospy.get_param('~model_path'))
        self.linear_model.to(self.device)
        self.linear_model.eval()
        rospy.loginfo("loading pytorch linear model end")

        self.bridge = cv_bridge.CvBridge()
        self.pub_rects = self.advertise(
            '~output/rects', RectArray, queue_size=1)
        self.pub_class = self.advertise(
            '~output/class', ClassificationResult, queue_size=1)

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 30)
        sub_img = message_filters.Subscriber(
            '~input',
            Image, queue_size=1, buff_size=2**24)
        sub_seg = message_filters.Subscriber(
            '~input/segmentation',
            Image, queue_size=1, buff_size=2**24)
        self.subs = [sub_img, sub_seg]

        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self.callback)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def callback(self, img_msg, seg_msg):
        bridge = self.bridge

        img = bridge.imgmsg_to_cv2(img_msg, 'rgb8')
        label_instance = bridge.imgmsg_to_cv2(seg_msg, 'passthrough')
        instances = np.unique(label_instance)
        instances = instances[instances != -1]
        n_instance = len(instances)
        boxes = np.zeros((n_instance, 4), dtype=np.int32)
        labels = []
        label_names = []
        scores = []

        features = []
        for i, inst in enumerate(instances):
            mask_inst = label_instance == inst
            where = np.argwhere(mask_inst)
            (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
            boxes[i] = (x1, y1, x2 - x1, y2 - y1)
            features.append(self.clip_model.encode_image(self.preprocess(
                cv2pil(img[y1:y2, x1:x2])).unsqueeze(0).to(self.device)))

        if len(features) > 0:
            with torch.no_grad():
                y = self.linear_model(torch.cat(features).to(torch.float32).to(self.device))
                preds = torch.argmax(y, 1).cpu().numpy()
                labels = preds.tolist()
                scores = torch.max(y, 1).values.cpu().numpy().tolist()
                label_names = list(map(str, labels))
        else:
            labels = []
            scores = []
            label_names = []

        self.pub_rects.publish(boxes_to_rectsarray(boxes, header=img_msg.header))
        self.pub_class.publish(
                ClassificationResult(
                    header=img_msg.header,
                    classifier='clip',
                    labels=labels,
                    label_names=label_names,
                    label_proba=scores,
                )
            )


if __name__ == '__main__':
    rospy.init_node('clip_classifier_node')
    node = ClipClassifierNode()
    rospy.spin()
