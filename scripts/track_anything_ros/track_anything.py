from track_anything_ros.segmentator.sam_segmentator import SAMSegmentator
from track_anything_ros.tracker.base_tracker import BaseTracker


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
