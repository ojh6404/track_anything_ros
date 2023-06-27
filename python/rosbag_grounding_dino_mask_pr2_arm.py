#!/usr/bin/env python3

import time
import datetime
import os
import argparse

import rosbag
import rospkg
from cv_bridge import CvBridge

import numpy as np
import torch
import cv2
from PIL import Image
from moviepy.editor import ImageSequenceClip

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

from track_anything_ros.tracker.base_tracker import BaseTracker
from track_anything_ros.utils.util import (
    download_checkpoint,
    download_checkpoint_from_google_drive,
)
from segment_anything import sam_model_registry, SamPredictor

color_category = [67, 102, 28, 114, 3]
color_v_bias = [-20, -40, 0, 35, 15]

package_path = rospkg.RosPack().get_path("track_anything_ros")
SAM_CHECKPOINT = {
    "vit_h": os.path.join(package_path, "checkpoints/sam_vit_h_4b8939.pth"),
    "vit_b": os.path.join(package_path,"checkpoints/sam_vit_b_01ec64.pth")
}
XMEM_CHECKPOINT = "XMem-s012.pth"
XMEM_CHECKPOINT_URL = (
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
)


def cv2pil(image):
    """OpenCV型 -> PIL型"""
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def pil2cv(image):
    """PIL型 -> OpenCV型"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image


def load_image(image_cv2):
    # load image
    image_pil = cv2pil(image_cv2)  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_grounding_dino_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


class SegmentAndTrack(object):
    def __init__(
        self, model_dir, tracker_config, color_idx, model_type="vit_h", device="cuda"
    ):
        self.device = torch.device(device)
        self.model_type = model_type
        self.sam_predictor = self.init_segment_anything()
        self.color_idx = color_idx
        self.tracker_config_file = tracker_config

        self.xmem_checkpoint = download_checkpoint(
            XMEM_CHECKPOINT_URL, model_dir, XMEM_CHECKPOINT
        )
        self.xmem = BaseTracker(
            self.xmem_checkpoint, self.tracker_config_file, device=self.device
        )

    def init_segment_anything(self):
        """
        Initialize the segmenting anything with model_type in ['vit_b', 'vit_l', 'vit_h']
        """
        sam = sam_model_registry[self.model_type](
            checkpoint=SAM_CHECKPOINT[self.model_type]
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

        color = color_category[self.color_idx]
        v_bias = color_v_bias[self.color_idx]
        h_deg = (
            np.ones(len(mask_unique)) * color * np.random.uniform(low=0.95, high=1.05)
        )  # blue
        # s_deg = np.ones(len(mask_unique)) * 68
        # v_deg = np.ones(len(mask_unique)) * 140

        # h_deg = [0, 100]
        s_mag = np.random.uniform(low=0.9, high=1.1)
        v_mag = np.random.uniform(low=0.9, high=1.2)

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
                    randomized_frame[:, :, 2] * v_mag + v_bias, 0, 255
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
    
    def frames_to_mask_mono8(self, masks, frames_rgb):
        """
        Track the object from the template mask
        """
        frame_len = len(frames_rgb)
        # extracted_image = frames_rgb.copy()
        result_frames = []

        
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

    def rewrite_bag(self, new_bag_path, frames_mono8, topic_name):
        cnt = 0
        with rosbag.Bag(new_bag_path, "w") as outbag:
            for topic, msg, t in rosbag.Bag(self.bag_path).read_messages():
                if topic == topic_name:
                    header = msg.header
                    msg = CvBridge().cv2_to_compressed_imgmsg(frames_mono8[cnt])
                    msg.header = header
                    msg.format = "mono8; jpeg compressed "
                    cnt += 1
                    topic = "/track_anything/mask_image"
                outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)

    def get_first_bgr(self, topic_name):
        for topic, msg, _ in self.bag.read_messages():
            if topic == topic_name:
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
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--bag_dir", type=str, required=True, help="rosbag directory")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--num_aug", type=int, default=1, help="number of aug")
    parser.add_argument(
        "--box_threshold", type=float, default=0.3, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, help="text threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cpu or cuda",
    )
    args = parser.parse_args()

    # cfg
    bag_dir = args.bag_dir
    text_prompt = args.text_prompt
    num_aug = args.num_aug
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    
    config_file = os.path.join(package_path, "config","GroundingDINO_SwinT_OGC.py")
    tracker_config = os.path.join(package_path, "config","tracker_config.yaml")
    grounded_checkpoint = os.path.join(
        package_path, "checkpoints","groundingdino_swint_ogc.pth"
    )
    sam_checkpoint = os.path.join(package_path, "checkpoints","sam_vit_h_4b8939.pth")
    topic_name = "/kinect_head/rgb/image_rect_color/compressed"
    sam_model = "vit_h"
    model_dir = os.path.join(package_path, "checkpoints")
    bags = get_bag_files_from_dir(bag_dir)
    bags.sort()  # sort by name


    grounding_dino_model = load_grounding_dino_model(config_file, grounded_checkpoint, device=device)
    
    for idx in range(len(color_category)):
        for bag_path in bags:
            bag_file = bag_path.split("/")[-1]

            bag_handler = BagHandler(bag_path)
            first_bgr = bag_handler.get_first_bgr(topic_name=topic_name)

            image_pil, image = load_image(first_bgr)
            boxes_filt, pred_phrases = get_grounding_output(
                grounding_dino_model, image, text_prompt, box_threshold, text_threshold, device=device
            )

            first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
            segment_and_track = SegmentAndTrack(
                model_dir, tracker_config, idx, sam_model
            )
            segment_and_track.sam_predictor.set_image(first_rgb)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = segment_and_track.sam_predictor.transform.apply_boxes_torch(
                boxes_filt, first_rgb.shape[:2]
            ).to(device)

            if transformed_boxes.size(0) == 0:
                continue

            masks, _, _ = segment_and_track.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(device),
                multimask_output=False,
            )

            masks = masks.squeeze(1).cpu().numpy()
            template_mask = generate_multi_mask(masks)
            frames_rgb, stamps, times = bag_handler.get_frames_from_bag(
                topic=topic_name
            )
            masks, painted_frames = segment_and_track.track_from_mask(
                template_mask=template_mask, frames_rgb=frames_rgb
            )
            
            processed_masks = []
            masks_for_gif = []
            for mask in masks:
                processed_masks.append(np.clip(mask * 255, 0, 255).astype(np.uint8))
                masks_for_gif.append(np.clip(cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB), 0, 255).astype(np.uint8))

            
            # mask_frames_mono8 = segment_and_track.frames_to_mask_mono8(masks)
            time_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # save numpy array randomized_frames to gif
            clip = ImageSequenceClip(masks_for_gif, fps=30)
            clip.write_gif(
                package_path + "/python/mask_gif/randomized_frames-" + time_now + ".gif"
            )

            # save rosbag
            save_filename = "train-episode-" + time_now + ".bag"
            bag_handler.rewrite_bag(
                package_path + "/python/mask_output/" + save_filename, processed_masks, topic_name=topic_name
            )
            print("bag file saved to", package_path + "/python/mask_output/" + save_filename)
