from tqdm import tqdm

from track_anything_ros.segmentator.sam_segmentator import SAMSegmentator

from track_anything_ros.tracker.base_tracker import BaseTracker

# from track_anything_ros.inpainter.base_inpainter import BaseInpainter
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

    # def first_frame_click(
    #     self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True
    # ):
    #     mask, logit, painted_image = self.sam.process_prompt(
    #         image, points, labels, multimask
    #     )
    #     return mask, logit, painted_image

    # def generator(self, images: list, template_mask: np.ndarray):
    #     masks = []
    #     logits = []
    #     painted_images = []
    #     for i in tqdm(range(len(images)), desc="Tracking image"):
    #         if i == 0:  # when first frame
    #             mask, logit, painted_image = self.xmem.track(
    #                 frame=images[i], first_frame_annotation=template_mask
    #             )
    #             masks.append(mask)
    #             logits.append(logit)
    #             painted_images.append(painted_image)
    #         else:
    #             mask, logit, painted_image = self.xmem.track(images[i])
    #             masks.append(mask)
    #             logits.append(logit)
    #             painted_images.append(painted_image)
    #     return masks, logits, painted_images
