from tqdm import tqdm

from track_anything_ros.segmentator.sam_segmentator import SAMSegmentator
from track_anything_ros.tracker.base_tracker import BaseTracker
from track_anything_ros.inpainter.base_inpainter import BaseInpainter
import numpy as np


class TrackAnything(object):
    def __init__(
        self, sam_checkpoint, sam_model_type, xmem_checkpoint, e2fgvi_checkpoint, device
    ):
        # self.args = args  # device, model type ...
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.e2fgvi_checkpoint = e2fgvi_checkpoint
        self.sam_model_type = sam_model_type
        self.device = device
        self.sam = SAMSegmentator(self.sam_checkpoint, self.sam_model_type, self.device)
        self.xmem = BaseTracker(self.xmem_checkpoint, device=self.device)
        # TODO
        # self.baseinpainter = BaseInpainter(self.e2fgvi_checkpoint, args.device)

        self.prompt = None

    def first_frame_click(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True
    ):
        mask, logit, painted_image = self.sam.first_frame_click(
            image, points, labels, multimask
        )
        return mask, logit, painted_image

    def generator(self, images: list, template_mask: np.ndarray):
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i == 0:  # when first frame
                mask, logit, painted_image = self.xmem.track(
                    frame=images[i], first_frame_annotation=template_mask
                )
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images

    # use sam to get the mask
    def sam_refine(
        self, video_state, point_prompt, click_state, interactive_state, point_coord
    ):
        """
        Args:
            template_frame: PIL.Image
            point_prompt: flag for positive or negative button click
            click_state: [[points], [labels]]
        """
        if point_prompt == "Positive":
            coordinate = "[[{},{},1]]".format(point_coord[0], point_coord[1])
            interactive_state["positive_click_times"] += 1
        else:
            coordinate = "[[{},{},0]]".format(point_coord[0], point_coord[1])
            interactive_state["negative_click_times"] += 1

        # prompt for sam model
        model.samcontroler.sam_controler.reset_image()
        model.samcontroler.sam_controler.set_image(
            video_state["origin_images"][video_state["select_frame_number"]]
        )
        prompt = get_prompt(click_state=click_state, click_input=coordinate)

        mask, logit, painted_image = model.first_frame_click(
            image=video_state["origin_images"][video_state["select_frame_number"]],
            points=np.array(prompt["input_point"]),
            labels=np.array(prompt["input_label"]),
            multimask=prompt["multimask_output"],
        )
        video_state["masks"][video_state["select_frame_number"]] = mask
        video_state["logits"][video_state["select_frame_number"]] = logit
        video_state["painted_images"][
            video_state["select_frame_number"]
        ] = painted_image

        operation_log = [
            ("", ""),
            (
                "Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment",
                "Normal",
            ),
        ]
        return painted_image, video_state, interactive_state, operation_log
