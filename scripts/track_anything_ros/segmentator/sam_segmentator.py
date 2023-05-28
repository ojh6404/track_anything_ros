from PIL import Image
import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor
from track_anything_ros.utils.painter import mask_painter, point_painter

mask_color = 3
mask_alpha = 0.7
contour_color = 1
contour_width = 5
point_color_ne = 8
point_color_ps = 50
point_alpha = 0.9
point_radius = 15
contour_color = 2
contour_width = 5


class SAMSegmentator(object):
    def __init__(self, SAM_checkpoint, model_type, device="cuda:0"):
        """
        initialize SAM segmentator
        """
        assert model_type in [
            "vit_b",
            "vit_l",
            "vit_h",
        ], "model_type must be vit_b, vit_l, or vit_h"

        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=SAM_checkpoint)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
        self.embedded_image = None  #

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # image embedding: avoid encode the same image multiple times
        self.embedded_image = image
        # print("why image ", image)
        if self.embedded_image is not None:
            print("repeat embedding, please reset_image.")
            return
        self.predictor.set_image(image)
        return

    @torch.no_grad()
    def reset_image(self):
        self.predictor.reset_image()
        self.embedded_image = None

    def predict(self, prompts, mode, multimask=True):
        """
        image: numpy array, h, w, 3
        prompts: dictionary, 3 keys: 'point_coords', 'point_labels', 'mask_input'
        prompts['point_coords']: numpy array [N,2]
        prompts['point_labels']: numpy array [1,N]
        prompts['mask_input']: numpy array [1,256,256]
        mode: 'point' (points only), 'mask' (mask only), 'both' (consider both)
        mask_outputs: True (return 3 masks), False (return 1 mask only)
        whem mask_outputs=True, mask_input=logits[np.argmax(scores), :, :][None, :, :]
        """
        assert (
            self.embedded_image is not None
        ), "prediction is called before set_image (feature embedding)."
        assert mode in ["point", "mask", "both"], "mode must be point, mask, or both"

        if mode == "point":
            masks, scores, logits = self.predictor.predict(
                point_coords=prompts["point_coords"],
                point_labels=prompts["point_labels"],
                multimask_output=multimask,
            )
        elif mode == "mask":
            masks, scores, logits = self.predictor.predict(
                mask_input=prompts["mask_input"], multimask_output=multimask
            )
        elif mode == "both":  # both
            masks, scores, logits = self.predictor.predict(
                point_coords=prompts["point_coords"],
                point_labels=prompts["point_labels"],
                mask_input=prompts["mask_input"],
                multimask_output=multimask,
            )
        else:
            raise NotImplementedError
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks, scores, logits

    def process_prompt(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        multimask=True,
        mask_color=3,
    ):
        """
        it is used in first frame in video
        return: mask, logit, painted image(mask+point)
        """
        label_flag = labels[-1]
        if label_flag == 1:
            # positive
            prompts = {
                "point_coords": points,
                "point_labels": labels,
            }
            masks, scores, logits = self.predict(prompts, "point", multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            prompts = {
                "point_coords": points,
                "point_labels": labels,
                "mask_input": logit[None, :, :],
            }
            masks, scores, logits = self.predict(prompts, "both", multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        else:
            # negative
            prompts = {
                "point_coords": points,
                "point_labels": labels,
            }
            masks, scores, logits = self.predict(prompts, "point", multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

        assert len(points) == len(labels)

        painted_image = mask_painter(
            image,
            mask.astype("uint8"),
            mask_color,
            mask_alpha,
            contour_color,
            contour_width,
        )
        # positive label marker
        painted_image = point_painter(
            painted_image,
            np.squeeze(points[np.argwhere(labels > 0)], axis=1),
            point_color_ne,
            point_alpha,
            point_radius,
            contour_color,
            contour_width,
        )
        # negative label marker
        painted_image = point_painter(
            painted_image,
            np.squeeze(points[np.argwhere(labels < 1)], axis=1),
            point_color_ps,
            point_alpha,
            point_radius,
            contour_color,
            contour_width,
        )
        painted_image = Image.fromarray(painted_image)
        return mask, logit, painted_image
