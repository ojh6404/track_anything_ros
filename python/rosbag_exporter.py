#!/usr/bin/env python3

from typing import Any
import rosbag
from moviepy.editor import ImageSequenceClip

# import cv_bridge
from cv_bridge import CvBridge
import numpy as np
import rospkg
import time
import datetime

# Install detectron2
import torch
import sys
import numpy as np
import os, json, cv2, random

import track_anything_ros
from track_anything_ros.tracker.base_tracker import BaseTracker
from track_anything_ros.utils.util import (
    download_checkpoint,
    download_checkpoint_from_google_drive,
)


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from segment_anything import sam_model_registry, SamPredictor

setup_logger()

# import some common libraries

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# Detic libraries
sys.path.insert(
    0,
    "/home/leus/ros/pr2_ws/src/track_anything_ros/python/Detic/third_party/CenterNet2/",
)
sys.path.insert(0, "/home/leus/ros/pr2_ws/src/track_anything_ros/python/Detic")
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder


models = {
    "vit_h": "/home/leus/ros/pr2_ws/src/track_anything_ros/checkpoints/sam_vit_h_4b8939.pth",
    "vit_b": "/home/leus/ros/pr2_ws/src/track_anything_ros/checkpoints/sam_vit_b_01ec64.pth",
}

XMEM_CHECKPOINT = "XMem-s012.pth"
XMEM_CHECKPOINT_URL = (
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
)


BUILDIN_CLASSIFIER = {
    "lvis": "/home/leus/.mohou/pr2_test/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy",
    "objects365": "/home/leus/.mohou/pr2_test/Detic/datasets/metadata/o365_clip_a+cnamefix.npy",
    "openimages": "/home/leus/.mohou/pr2_test/Detic/datasets/metadata/oid_clip_a+cname.npy",
    "coco": "/home/leus/.mohou/pr2_test/Detic/datasets/metadata/coco_clip_a+cname.npy",
}

BUILDIN_METADATA_PATH = {
    "lvis": "lvis_v1_val",
    "objects365": "objects365_v2_val",
    "openimages": "oid_val_expanded",
    "coco": "coco_2017_val",
}


class DeticPredictor(object):
    def __init__(
        self,
        cfg_file,
        model_weight,
        device="cuda",
        vocabulary="custom",
        custom_vocabulary=None,
    ):
        self.cfg = get_cfg()
        add_centernet_config(self.cfg)
        add_detic_config(self.cfg)
        self.device = torch.device(device)

        self.cfg.merge_from_file(cfg_file)
        self.cfg.MODEL.WEIGHTS = model_weight
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        self.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (
            True  # For better visualization purpose. Set to False for all classes.
        )
        # cfg.MODEL.DEVICE= device # uncomment this to use cpu-only mode.
        self.predictor = DefaultPredictor(self.cfg)

        assert vocabulary in ["lvis", "objects365", "openimages", "coco", "custom"]
        if vocabulary == "custom":
            self.vocabulary = vocabulary
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = (
                custom_vocabulary  # Change here to try your own vocabularies!
            )
            self.classifier = self.get_clip_embeddings(self.metadata.thing_classes)
            self.num_classes = len(self.metadata.thing_classes)
            reset_cls_test(self.predictor.model, self.classifier, self.num_classes)
        else:
            self.vocabulary = (
                vocabulary  # change to 'lvis', 'objects365', 'openimages', or 'coco'
            )
            self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[self.vocabulary])
            self.classifier = BUILDIN_CLASSIFIER[self.vocabulary]
            self.num_classes = len(self.metadata.thing_classes)
            reset_cls_test(self.predictor.model, self.classifier, self.num_classes)

        # Reset visualization threshold
        self.output_score_threshold = 0.3
        for cascade_stages in range(len(self.predictor.model.roi_heads.box_predictor)):
            self.predictor.model.roi_heads.box_predictor[
                cascade_stages
            ].test_score_thresh = self.output_score_threshold

    def get_clip_embeddings(self, vocabulary, prompt="a "):
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        texts = [prompt + x for x in vocabulary]
        emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    def __call__(self, image):
        return self.predictor(image)


class SegmentAndTrack(object):
    def __init__(self, model_dir, tracker_config, model_type="vit_h", device="cuda"):
        self.device = torch.device(device)
        self.model_type = model_type
        self.sam_predictor = self.init_segment_anything()

        self.tracker_config_file = (
            "/home/leus/ros/pr2_ws/src/track_anything_ros/config/tracker_config.yaml"
        )

        self.xmem_checkpoint = download_checkpoint(
            XMEM_CHECKPOINT_URL, model_dir, XMEM_CHECKPOINT
        )
        self.xmem = BaseTracker(
            self.xmem_checkpoint, self.tracker_config_file, device=self.device
        )

        # self.image = None
        # template_mask = self.generate_multi_mask(
        #     masks
        # )  # masks [N, H, W] to template_mask [H, W]
        # self.mask, self.logit, self.painted_image = self.xmem.track(
        #     frame=self.image, first_frame_annotation=self.template_mask
        # )

        # self.xmem.clear_memory()

    def init_segment_anything(self):
        """
        Initialize the segmenting anything with model_type in ['vit_b', 'vit_l', 'vit_h']
        """
        sam = sam_model_registry[self.model_type](
            checkpoint=models[self.model_type]
        ).to(self.device)
        predictor = SamPredictor(sam)
        return predictor

    def track_from_mask(self, template_mask, frames_rgb):
        """
        Track the object from the template mask
        """
        self.xmem.clear_memory()
        frame_len = len(frames_rgb)
        painted_images = []
        masks = []
        for i in range(frame_len):
            frame = frames_rgb[i]
            if i == 0:
                mask, logit, painted_image = self.xmem.track(
                    frame=frame, first_frame_annotation=template_mask
                )
            else:
                mask, logit, painted_image = self.xmem.track(frame)

            painted_images.append(painted_image)
            masks.append(mask)

        self.xmem.clear_memory()
        return masks, painted_images

    def randomize_frames(self, masks, frames_rgb):
        """
        Track the object from the template mask
        """
        frame_len = len(frames_rgb)
        # extracted_image = frames_rgb.copy()
        result_frames = []

        mask_unique = np.unique(masks[0])
        np.random.seed(int(time.time()))
        h_deg_max = 180
        h_deg = np.random.uniform(low=0, high=h_deg_max, size=len(mask_unique))

        h_deg = (
            np.ones(len(mask_unique)) * 100 * np.random.uniform(low=0.98, high=1.02)
        )  # blue
        # s_deg = np.ones(len(mask_unique)) * 68
        # v_deg = np.ones(len(mask_unique)) * 140

        # h_deg = [0, 100]
        s_mag = np.random.uniform(low=0.8, high=1.2)
        v_mag = np.random.uniform(low=0.8, high=1.2)

        for i in range(frame_len):
            frame_rgb = frames_rgb[i]
            mask = masks[i]
            result_frame = frame_rgb.copy()

            randomized_frames = []
            for mask_idx in range(1, len(np.unique(mask))):
                randomized_frame = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2HSV)

                # hue = randomized_frame[:, :, 0]
                # offset_hue = hue - 90
                # randomized_hue = (hue + h_deg[mask_idx - 1]) % h_deg_max
                # randomized_frame[:, :, 0] = np.clip(randomized_hue ,0, h_deg_max)
                # TODO

                randomized_frame[:, :, 0] = np.clip(
                    h_deg[mask_idx - 1], 0, h_deg_max
                ).astype(np.uint8)
                randomized_frame[:, :, 1] = np.clip(
                    randomized_frame[:, :, 1] * s_mag, 0, 255
                ).astype(np.uint8)
                randomized_frame[:, :, 2] = np.clip(
                    randomized_frame[:, :, 2] * v_mag, 0, 255
                ).astype(np.uint8)
                # randomized_frame[:, :, 0] = np.clip(h_deg[mask_idx -    1], 0, h_deg_max)
                # randomized_frame[:, :, 1] = np.clip(s_deg[mask_idx - 1] * s_mag, 0, 255)
                # randomized_frame[:, :, 2] = np.clip(v_deg[mask_idx - 1] * v_mag, 0, 255)

                randomized_frame = cv2.cvtColor(randomized_frame, cv2.COLOR_HSV2RGB)
                randomized_frames.append(randomized_frame)

            for mask_idx in range(1, len(np.unique(mask))):
                for channel_idx in range(frame_rgb.shape[2]):
                    result_frame[:, :, channel_idx] = np.where(
                        mask == mask_idx,
                        randomized_frames[mask_idx - 1][:, :, channel_idx],
                        result_frame[:, :, channel_idx],
                    )
            result_frames.append(result_frame)
        return result_frames

    def generate_multi_mask(self, masks):
        template_mask = np.zeros_like(masks[0])
        # for i in range(1, len(self.masks)):
        for i, mask in enumerate(masks):
            template_mask = np.clip(
                template_mask + mask * (i + 1),
                0,
                i + 1,
            )

        assert len(np.unique(template_mask)) == (len(masks) + 1)
        return template_mask


class BagHandler(object):
    def __init__(self, bag_path):
        self.bag_path = bag_path
        self.bag = rosbag.Bag(bag_path)
        self.topics, self.types = self.get_topics()

    def get_topics(self):
        topics = self.bag.get_type_and_topic_info()[1].keys()
        types = []
        for val in self.bag.get_type_and_topic_info()[1].values():
            types.append(val[0])
        return topics, types

    def get_frames_from_bag(self, topic):
        frames = []
        stamps = []
        times = []
        for topic, msg, t in rosbag.Bag(self.bag_path).read_messages(topics=topic):
            frame_bgr = CvBridge().compressed_imgmsg_to_cv2(msg)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            stamps.append(msg.header.stamp)
            times.append(t)
        return frames, stamps, times

    def rewrite_bag(self, new_bag_path, frames_rgb):
        cnt = 0
        with rosbag.Bag(new_bag_path, "w") as outbag:
            for topic, msg, t in rosbag.Bag(self.bag_path).read_messages():
                if topic == "/kinect_head/rgb/image_rect_color/compressed":
                    frames_bgr = cv2.cvtColor(frames_rgb[cnt], cv2.COLOR_RGB2BGR)
                    header = msg.header
                    format = msg.format
                    msg = CvBridge().cv2_to_compressed_imgmsg(frames_bgr)
                    msg.header = header
                    msg.format = format
                    cnt += 1
                outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)

    def get_first_bgr(self):
        for topic, msg, _ in self.bag.read_messages():
            if topic == "/kinect_head/rgb/image_rect_color/compressed":
                self.bag.close()
                first_bgr = CvBridge().compressed_imgmsg_to_cv2(msg)
                return first_bgr
        self.bag.close()
        assert False


def generate_multi_mask(masks):
    template_mask = np.zeros_like(masks[0])
    # for i in range(1, len(self.masks)):
    for i, mask in enumerate(masks):
        template_mask = np.clip(
            template_mask + mask * (i + 1),
            0,
            i + 1,
        )

    assert len(np.unique(template_mask)) == (len(masks) + 1)
    return template_mask


def get_bag_files_from_dir(dir):
    bag_files = []
    for file in os.listdir(dir):
        if file.endswith(".bag"):
            bag_files.append(os.path.join(dir, file))
    return bag_files


if __name__ == "__main__":
    """
    detic works for cv2 bgr image
    sam works for rgb
    detic for bboxes detection
    sam for segmentation from boxes
    xmem for tracking
    """

    package_path = rospkg.RosPack().get_path("track_anything_ros")
    # bag_path = package_path + "/python/rosbag/train2.bag"
    # bag_dir = package_path + "/python/rosbag"
    bag_dir = "/home/leus/.mohou/pr2_track.backup/rosbag"
    device = "cuda"
    sam_model = "vit_h"
    cfg_file = (
        package_path
        + "/python/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
    )
    model_weight = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    custom_vocabulary = [
        "dish",
        # "pr2 robot arm",
        "bowl",
    ]
    # custom_vocabulary = ["bowl",]
    tracker_config = (
        "/home/leus/ros/pr2_ws/src/track_anything_ros/config/tracker_config.yaml"
    )
    model_dir = package_path + "/python/checkpoints"

    bags = get_bag_files_from_dir(bag_dir)
    bags.sort()  # sort by name
    print("bags", bags)
    print("file name of bag", [filename.split("/")[-1] for filename in bags])

    num_aug = 1

    for _ in range(num_aug):
        for bag_path in bags:
            bag_file = bag_path.split("/")[-1]

            bag_handler = BagHandler(bag_path)
            first_bgr = bag_handler.get_first_bgr()

            detic_predictor = DeticPredictor(
                cfg_file,
                model_weight,
                device="cuda:0",
                vocabulary="custom",
                custom_vocabulary=custom_vocabulary,
            )
            outputs = detic_predictor(first_bgr)

            print(
                "predicted classes :",
                [
                    detic_predictor.metadata.thing_classes[x]
                    for x in outputs["instances"].pred_classes.cpu().tolist()
                ],
            )  # class names
            if outputs["instances"].pred_classes.cpu().tolist() == []:
                continue
            scores = outputs["instances"].scores.cpu().numpy()  # [N]
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()  # [N, 4]
            masks = outputs["instances"].pred_masks.cpu().numpy()  # [N, H, W]

            # get mask from bouding boxes with segment anything
            segment_and_track = SegmentAndTrack(model_dir, tracker_config, sam_model)
            first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
            segment_and_track.sam_predictor.set_image(first_rgb)

            boxes_tensor = torch.tensor(
                boxes.astype(int), device=device
            )  # [N, 1, H, W]
            transformed_boxes = (
                segment_and_track.sam_predictor.transform.apply_boxes_torch(
                    boxes_tensor, first_rgb.shape[:2]
                )
            )

            masks, _, _ = segment_and_track.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            masks = masks.squeeze(1).cpu().numpy()
            template_mask = generate_multi_mask(masks)

            frames_rgb, stamps, times = bag_handler.get_frames_from_bag(
                "/kinect_head/rgb/image_rect_color/compressed"
            )
            masks, painted_frames = segment_and_track.track_from_mask(
                template_mask=template_mask, frames_rgb=frames_rgb
            )
            randomized_frames = segment_and_track.randomize_frames(masks, frames_rgb)
            time_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # save numpy array randomized_frames to gif
            clip = ImageSequenceClip(randomized_frames, fps=30)
            clip.write_gif(
                package_path + "/python/test/randomized_frames-" + time_now + ".gif"
            )

            # for i, frame in enumerate(randomized_frames):
            #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(package_path + "/python/test/frame_{}.png".format(i), frame)

            save_filename = "train-episode-" + time_now + ".bag"

            bag_handler.rewrite_bag(
                package_path + "/python/output/" + save_filename, randomized_frames
            )
            print("bag file saved to", package_path + "/python/output/" + save_filename)
