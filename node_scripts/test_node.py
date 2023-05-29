#!/usr/bin/env python3
# import base64

import cv2
import cv_bridge
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import ClusterPointIndices
from jsk_recognition_msgs.msg import Label
from jsk_recognition_msgs.msg import LabelArray
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from jsk_topic_tools import ConnectionBasedTransport
import numpy as np
from PIL import Image
import torch

# from pcl_msgs.msg import PointIndices
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped

from track_anything_ros.track_anything import TrackAnything
from track_anything_ros.utils import util as TrackAnythingUtil
from track_anything_ros.utils.painter import point_painter, mask_painter

import argparse
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse


# from track_anything_ros.XMem.inference.data.test_datasets import (
#     LongTestDataset,
#     DAVISTestDataset,
#     YouTubeVOSTestDataset,
# )
# from track_anything_ros.XMem.inference.data.mask_mapper import MaskMapper
# from track_anything_ros.XMem.model.network import XMem
# from track_anything_ros.XMem.inference.inference_core import InferenceCore

# from track_anything_ros.XMem.inference.interact.interactive_utils import (
#     image_to_torch,
#     index_numpy_to_one_hot_torch,
#     torch_prob_to_numpy_mask,
#     overlay_davis,
# )

# from progressbar import progressbar
#
model_dir = "/home/leus/ros/catkin_ws/src/track_anything_ros/scripts/track_anything_ros/checkpoints/"
args = dict()
args["sam_checkpoint"] = model_dir + "sam_vit_b_01ec64.pth"
args["xmem_checkpoint"] = model_dir + "XMem-s012.pth"
args["sam_model_type"] = "vit_b"
args["device"] = "cuda:0"
args["e2fgvi_checkpoint"] = model_dir + "E2FGVI-HQ-CVPR22.pth"

# torch.set_grad_enabled(False)

# default configuration
# config = {
#     "top_k": 30,
#     "mem_every": 5,
#     "deep_update_every": -1,
#     "enable_long_term": True,
#     "enable_long_term_count_usage": True,
#     "num_prototypes": 128,
#     "min_mid_term_frames": 5,
#     "max_mid_term_frames": 10,
#     "max_long_term_elements": 10000,
# }

# network = XMem(config, args["xmem_checkpoint"]).eval().to("cuda:0")


mask_color = 3
mask_alpha = 0.7
contour_color = 1
# contour_width = 5
contour_width = 3
point_color_ne = 8
point_color_ps = 50
point_alpha = 0.9
# point_radius = 15
point_radius = 10
contour_color = 2
contour_width = 3


class InstanceSegmentationNode(ConnectionBasedTransport):
    def __init__(self, args):
        super(InstanceSegmentationNode, self).__init__()

        self.track_anything = TrackAnything(
            sam_checkpoint=args["sam_checkpoint"],
            sam_model_type=args["sam_model_type"],
            xmem_checkpoint=args["xmem_checkpoint"],
            e2fgvi_checkpoint=args["e2fgvi_checkpoint"],
            device=args["device"],
        )
        self.device = args["device"]

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

        # self.prompt = dict()
        # self.points = [[[350, 290]]]
        # self.points = [[350, 290]]
        self.points = []
        self.labels = []  # List, (Num of Masks, Num of Points)
        # List, (Num of Masks, Num of points, 2)
        # self.labels = [[1]]  # List, (Num of Masks, Num of Points)
        self.multimask = True

        self.logits = []
        self.painted_image = []
        self.masks = []

        # for place holder init
        self.embedded_image = None  # initialized by None
        self.image = None
        self.track_image = None
        self.mask = None
        self.logit = None

        self.template_mask = None

        self.painted_image = None

        # self.xmem = InferenceCore(network, config=config)

    # @torch.no_grad()
    # def clear_memory(self):
    #     self.xmem.clear_memory()
    #     torch.cuda.empty_cache()
    # self.track_anything.xmem.clear_memory()

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
        self.track_anything.sam.set_image(self.image)
        res = EmptyResponse()
        return res

    def reset_embed_callback(self, srv):
        rospy.loginfo("Reset Embedding image")
        self.embedded_image = None
        self.track_anything.sam.reset_image()
        res = EmptyResponse()
        return res

    def track_trigger_callback(self, srv):
        rospy.loginfo("Tracking start...")
        self.template_mask = self.generate_multi_mask(self.masks)
        # num_objects = len(np.unique(self.template_mask)) - 1
        # self.xmem.set_all_labels(range(1, num_objects + 1))  # consecutive labels

        # self.frame_torch, _ = image_to_torch(self.image, device=self.device)
        # self.mask_torch = index_numpy_to_one_hot_torch(
        #     self.template_mask, num_objects + 1
        # ).to(self.device)
        # the background mask is not fed into the model
        # self.prediction = self.xmem.step(self.frame_torch, self.mask_torch[1:])
        self.mask, self.logit, self.painted_image = self.track_anything.xmem.track(
            frame=self.image, first_frame_annotation=self.template_mask
        )
        # self.prediction = torch_prob_to_numpy_mask(self.prediction)
        # self.painted_image = overlay_davis(self.image, self.prediction)
        res = EmptyResponse()
        return res

    def subscribe(self):
        # TODO
        self.sub_image = rospy.Subscriber(
            "~input",
            # "/zed/zed_node/rgb/image_rect_color",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.sub_point = rospy.Subscriber(
            "/kinect_head/rgb/image_color/screenpoint",
            PointStamped,
            self.point_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.subs = [self.sub_image, self.sub_point]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()
        self.track_anything.xmem.clear_memory()

    def point_callback(self, point_msg):
        # point_x = np.clip(int(point_msg.point.x), 0, self.image.shape[0])
        # point_y = np.clip(int(point_msg.point.y), 0, self.image.shape[1])

        # point = [point_x, point_y]
        point = [int(point_msg.point.x), int(point_msg.point.y)]
        label = 1  # TODO
        rospy.loginfo("point {} and label {} added".format(point, label))
        self.points.append(point)
        self.labels.append(label)

        if self.embedded_image is None:
            # self.track_anything.sam.reset_image()
            self.track_anything.sam.set_image(self.image)
            self.embedded_image = self.image

        self.mask, self.logit = self.track_anything.sam.process_prompt(
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
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        if self.template_mask is not None:  # track start
            # with torch.cuda.amp.autocast(enabled=True):
            #     self.frame_torch, _ = image_to_torch(self.image, device=self.device)
            #     self.prediction = self.xmem.step(self.frame_torch)
            #     self.prediction = torch_prob_to_numpy_mask(self.prediction)
            #     self.painted_image = overlay_davis(self.image, self.prediction)

            # self.clear_memory()
            self.mask, self.logit, self.painted_image = self.track_anything.xmem.track(
                self.image
            )
        else:  # init
            # get masks

            if self.mask is not None:
                self.painted_image = mask_painter(
                    self.painted_image,
                    self.mask.astype("uint8"),
                    20 + mask_color,
                    mask_alpha,
                    contour_color,
                    contour_width,
                )

            # if len(self.masks) > 0:
            for i, mask in enumerate(self.masks):
                self.painted_image = mask_painter(
                    self.painted_image,
                    mask.astype("uint8"),
                    i + mask_color,
                    mask_alpha,
                    contour_color,
                    contour_width,
                )

            self.painted_image = point_painter(
                self.image if self.painted_image is None else self.painted_image,
                np.array(self.points),
                len(self.masks) + point_color_ne,
                point_alpha,
                point_radius,
                contour_color,
                contour_width,
            )

        out_img_msg = self.bridge.cv2_to_imgmsg(self.painted_image, encoding="bgr8")
        out_img_msg.header = img_msg.header
        self.pub_segment_img.publish(out_img_msg)

        # self.painted_image = mask_painter(
        #     self.painted_image,
        #     self.mask.astype("uint8"),
        #     len(self.masks) + mask_color,
        #     mask_alpha,
        #     len(self.masks) + contour_color,
        #     contour_width,
        # )
        # positive label marker
        # self.painted_image = point_painter(
        #     self.painted_image,
        #     np.squeeze(
        #         np.array(self.points)[np.argwhere(np.array(self.labels) > 0)], axis=1
        #     ),
        #     len(self.masks) + point_color_ne,
        #     point_alpha,
        #     point_radius,
        #     len(self.masks) + contour_color,
        #     contour_width,
        # )
        # for i in range(len(self.points)):
        #     mask, logit, painted_image = self.track_anything.sam.process_prompt(
        #         image=img,
        #         points=self.points[i],
        #         labels=self.labels[i],
        #         multimask=self.multimask,
        #     )
        #     self.masks.append(mask)
        #     self.logits.append(logit)
        #     self.painted_image.append(painted_image)
        # # set xmem first_frame
        # self.mask, self.logit, self.painted_image = self.track_anything.xmem.track(
        #     frame=img, first_frame_annotation=self.template_mask
        # )

        # rgb_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("frame", rgb_frame)

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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--sam_model_type", type=str, default="vit_b")
    # parser.add_argument(
    #     "--sam_checkpoint", type=str, default=model_dir + "sam_vit_b_01ec64.pth"
    # )
    # parser.add_argument(
    #     "--xmem_checkpoint", type=str, default=model_dir + "XMem-s012.pth"
    # )
    # parser.add_argument(
    #     "--e2fgvi_checkpoint", type=str, default=model_dir + "E2FGVI-HQ-CVPR22.pth"
    # )
    # args = parser.parse_args()

    # capture = cv2.VideoCapture(0)

    # while True:
    #     ret, frame = capture.read()
    #     cv2.imshow("frame", frame)
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

    # capture.release()
    # cv2.destroyAllWindows()

    rospy.init_node("instance_segmentation")
    node = InstanceSegmentationNode(args)
    rospy.spin()
