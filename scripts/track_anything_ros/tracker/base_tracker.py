import os
import yaml
import numpy as np
import torch
from torchvision import transforms

from track_anything_ros.tracker.model.network import XMem
from track_anything_ros.tracker.inference.inference_core import InferenceCore
from track_anything_ros.tracker.util.mask_mapper import MaskMapper
from track_anything_ros.tracker.util.range_transform import im_normalization
from track_anything_ros.utils.painter import mask_painter


class BaseTracker(object):
    def __init__(self, xmem_checkpoint, config_file, device) -> None:
        """
        device: model device
        xmem_checkpoint: checkpoint of XMem model
        """
        # load configurations
        with open(
            config_file,
            "r",
        ) as stream:
            config = yaml.safe_load(stream)
        # TODO

        # initialise XMem
        network = XMem(config, xmem_checkpoint).to(device).eval()
        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        # data transformation
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                im_normalization,
            ]
        )
        self.device = device

        # changable properties
        self.mapper = MaskMapper()

    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Input:
        frames: numpy arrays (H, W, 3)

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """

        if first_frame_annotation is not None:  # first frame mask
            # initialisation
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = None
            labels = None
        # prepare inputs
        frame_tensor = self.im_transform(frame).to(self.device)
        # track one frame
        probs, _ = self.tracker.step(frame_tensor, mask, labels)  # logits 2 (bg fg) H W

        # convert to mask
        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

        final_mask = np.zeros_like(out_mask)

        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        num_objs = final_mask.max()
        painted_image = frame
        for obj in range(1, num_objs + 1):
            if np.max(final_mask == obj) == 0:
                continue
            painted_image = mask_painter(
                painted_image, (final_mask == obj).astype("uint8"), mask_color=obj + 1
            )

        return final_mask, final_mask, painted_image

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()
