#!/usr/bin/env python3

import rospy
import cv_bridge
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Empty, EmptyResponse
from jsk_topic_tools import ConnectionBasedTransport
from torch import device

from track_anything_ros.segmentator.sam_segmentator import SAMSegmentator
from track_anything_ros.tracker.base_tracker import BaseTracker
from track_anything_ros.utils.painter import point_painter, mask_painter
from track_anything_ros.utils.util import (
    download_checkpoint,
    download_checkpoint_from_google_drive,
)

MASK_COLOR = 3
MASK_ALPHA = 0.9
POINT_COLOR_N = 8
POINT_COLOR_P = 50
POINT_ALPHA = 0.9
POINT_RADIUS = 5
CONTOUR_COLOR = 1
CONTOUR_WIDTH = 0

SAM_CHECKPOINT_DICT = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
SAM_CHECKPOINT_URL_DICT = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
XMEM_CHECKPOINT = "XMem-s012.pth"
XMEM_CHECKPOINT_URL = (
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
)


class TrackNode(ConnectionBasedTransport):
    def __init__(self):
        super(TrackNode, self).__init__()

        model_dir = rospy.get_param("~model_dir")
        tracker_config_file = rospy.get_param("~tracker_config_file")
        # inpainter_config_file = rospy.get_param("~inpainter_config_file")
        model_type = rospy.get_param("~model_type", "vit_b")

        sam_checkpoint = download_checkpoint(
            SAM_CHECKPOINT_URL_DICT[model_type],
            model_dir,
            SAM_CHECKPOINT_DICT[model_type],
        )
        xmem_checkpoint = download_checkpoint(
            XMEM_CHECKPOINT_URL, model_dir, XMEM_CHECKPOINT
        )
        self.device = rospy.get_param("~device", "cuda:0")
        self.sam = SAMSegmentator(sam_checkpoint, model_type, device=self.device)
        self.xmem = BaseTracker(
            xmem_checkpoint, tracker_config_file, device=self.device
        )

        self.clear_points_service = rospy.Service(
            "/track_anything/clear_points", Empty, self.clear_points_callback
        )
        self.clear_masks_service = rospy.Service(
            "/track_anything/clear_masks", Empty, self.clear_masks_callback
        )
        self.add_mask_service = rospy.Service(
            "/track_anything/add_mask", Empty, self.add_mask_callback
        )
        self.set_embed_service = rospy.Service(
            "/track_anything/set_embed", Empty, self.set_embed_callback
        )
        self.reset_embed_service = rospy.Service(
            "/track_anything/reset_embed", Empty, self.reset_embed_callback
        )

        self.track_trigger_service = rospy.Service(
            "/track_anything/track_trigger", Empty, self.track_trigger_callback
        )

        self.pub_debug_image = self.advertise("~output_image", Image, queue_size=1)
        self.pub_segmentation_image = self.advertise(
            "~segmentation_mask", Image, queue_size=1
        )
        self.bridge = cv_bridge.CvBridge()

        self.points = []
        self.labels = []
        self.multimask = True

        self.logits = []
        self.painted_image = []
        self.masks = []

        # for place holder init
        self.embedded_image = None
        self.image = None
        self.track_image = None
        self.mask = None
        self.logit = None
        self.template_mask = None
        self.painted_image = None

    def clear_points_callback(self, srv):
        rospy.loginfo("Clear points")
        self.points.clear()
        self.labels.clear()
        self.mask = None
        self.logit = None
        self.painted_image = self.prev_painted_image.copy()
        res = EmptyResponse()
        return res

    def clear_masks_callback(self, srv):
        rospy.loginfo("Clear masks")
        self.masks.clear()
        self.mask = None
        self.logit = None
        self.painted_image = None
        res = EmptyResponse()
        return res

    def add_mask_callback(self, srv):
        rospy.loginfo("Mask added")

        res = EmptyResponse()
        if self.mask is None:
            rospy.logwarn("No mask to add")
            self.points.clear()
            self.labels.clear()
            return res

        self.masks.append(self.mask)
        self.points.clear()
        self.labels.clear()
        return res

    def set_embed_callback(self, srv):
        assert self.embedded_image is None, "reset before embedding"
        rospy.loginfo("Embedding image for segmentation")
        self.embedded_image = self.image
        self.sam.set_image(self.image)
        res = EmptyResponse()
        return res

    def reset_embed_callback(self, srv):
        rospy.loginfo("Reset Embedding image")
        self.embedded_image = None
        self.sam.reset_image()
        res = EmptyResponse()
        return res

    def track_trigger_callback(self, srv):
        rospy.loginfo("Tracking start...")
        self.template_mask = self.generate_multi_mask(self.masks)
        self.mask, self.logit, self.painted_image = self.xmem.track(
            frame=self.image, first_frame_annotation=self.template_mask
        )
        res = EmptyResponse()
        return res

    def subscribe(self):
        self.sub_image = rospy.Subscriber(
            "~input_image",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.sub_point = rospy.Subscriber(
            "~input_point",
            PointStamped,
            self.point_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.subs = [self.sub_image, self.sub_point]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()
        self.xmem.clear_memory()

    def point_callback(self, point_msg):
        # TODO: clipping point
        # point_x = np.clip(int(point_msg.point.x), 0, self.image.shape[0])
        # point_y = np.clip(int(point_msg.point.y), 0, self.image.shape[1])

        # point = [point_x, point_y]
        point = [int(point_msg.point.x), int(point_msg.point.y)]
        label = 1  # TODO: add negative label
        rospy.loginfo("point {} and label {} added".format(point, label))
        self.points.append(point)
        self.labels.append(label)

        if self.embedded_image is None:
            self.sam.set_image(self.image)
            self.embedded_image = self.image

        self.mask, self.logit = self.sam.process_prompt(
            image=self.image,
            points=np.array(self.points),
            labels=np.array(self.labels),
            multimask=self.multimask,
        )

        self.prev_painted_image = self.image.copy()

    def generate_multi_mask(self, masks):
        template_mask = np.zeros_like(masks[0])
        # for i in range(1, len(self.masks)):
        for i, mask in enumerate(masks):
            template_mask = np.clip(
                template_mask + mask * (i + 1),
                0,
                i + 1,
            )

        assert len(np.unique(template_mask)) == (len(self.masks) + 1)
        return template_mask

    def callback(self, img_msg):
        # self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")

        if self.template_mask is not None:  # track start
            self.mask, self.logit, self.painted_image = self.xmem.track(self.image)

            # encoding should be 32SC1 for jsk_pcl_utils/LabelToClusterPointIndices
            seg_mask = self.bridge.cv2_to_imgmsg(
                self.mask.astype(np.int32), encoding="32SC1"
            )
            # for debug
            # seg_mask = self.bridge.cv2_to_imgmsg(
            #     np.where(self.mask > 0, 255, 0).astype(np.uint8), encoding="mono8"
            # )
            # rospy.loginfo("segmented : {}".format(np.unique(self.mask)))
            seg_mask.header = img_msg.header
            self.pub_segmentation_image.publish(seg_mask)
        else:  # init
            if self.mask is not None:
                self.painted_image = mask_painter(
                    self.painted_image,
                    self.mask.astype("uint8"),
                    20 + MASK_COLOR,
                    MASK_ALPHA,
                    CONTOUR_COLOR,
                    CONTOUR_WIDTH,
                )

            # if len(self.masks) > 0:
            for i, mask in enumerate(self.masks):
                self.painted_image = mask_painter(
                    self.painted_image,
                    mask.astype("uint8"),
                    i + MASK_COLOR,
                    MASK_ALPHA,
                    CONTOUR_COLOR,
                    CONTOUR_WIDTH,
                )

            self.painted_image = point_painter(
                self.image if self.painted_image is None else self.painted_image,
                np.array(self.points),
                len(self.masks) + POINT_COLOR_N,
                POINT_ALPHA,
                POINT_RADIUS,
                CONTOUR_COLOR,
                CONTOUR_WIDTH,
            )

        # out_img_msg = self.bridge.cv2_to_imgmsg(self.painted_image, encoding="bgr8")
        out_img_msg = self.bridge.cv2_to_imgmsg(self.painted_image, encoding="rgb8")
        out_img_msg.header = img_msg.header
        self.pub_debug_image.publish(out_img_msg)


if __name__ == "__main__":
    rospy.init_node("track_node")
    node = TrackNode()
    rospy.spin()
