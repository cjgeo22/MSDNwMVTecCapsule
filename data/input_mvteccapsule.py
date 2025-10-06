# data/input_mvteccapsule.py
# Proper dataset entry for MVTec-AD "capsule", consuming our text splits directly.

import os
import numpy as np

from config import Config
from .input_ksdd2 import KSDD2Dataset  # we reuse its base utilities (to_tensor, read_label_resize, distance_transform, etc.)


class MVTecCapsuleDataset(KSDD2Dataset):
    """
    A clean adapter that reads images/masks using our text splits:

      DATASET_PATH/
        images/
        masks/
        splits/MVTecCapsule/{train.txt, segmented_train.txt, test.txt}

    The split files contain absolute (or repo-local absolute) *image* paths.
    We derive mask paths from the image basename using a few common patterns.
    """

    def __init__(self, kind: str, cfg: Config):
        # Make sure relative dirs exist on cfg (used by some utilities)
        if not getattr(cfg, "IMAGES_DIR", None):
            cfg.IMAGES_DIR = "images"
        if not getattr(cfg, "MASKS_DIR", None):
            cfg.MASKS_DIR = "masks"

        # Make sure split files are set (relative inside DATASET_PATH)
        if not getattr(cfg, "SPLIT_TRAIN", None):
            cfg.SPLIT_TRAIN = "splits/MVTecCapsule/train.txt"
        if not getattr(cfg, "SPLIT_SEGMENTED_TRAIN", None):
            cfg.SPLIT_SEGMENTED_TRAIN = "splits/MVTecCapsule/segmented_train.txt"
        if not getattr(cfg, "SPLIT_TEST", None):
            cfg.SPLIT_TEST = "splits/MVTecCapsule/test.txt"

        super().__init__(kind, cfg)  # this will call our read_contents()

    # ---- helpers ----------------------------------------------------------------

    def _split_paths(self):
        """Return list of image paths for current split, and set() of segmented ones (TRAIN only)."""
        base = os.path.join(self.cfg.DATASET_PATH, "splits", "MVTecCapsule")
        def _read_list(p):
            with open(p, "r") as f:
                return [ln.strip() for ln in f if ln.strip()]

        if str(self.kind).upper() == "TRAIN":
            train_txt = os.path.join(base, "train.txt")
            seg_txt   = os.path.join(base, "segmented_train.txt")
            train_list = _read_list(train_txt)
            seg_set = set(_read_list(seg_txt))
            return train_list, seg_set
        else:
            test_txt  = os.path.join(base, "test.txt")
            test_list = _read_list(test_txt)
            return test_list, set()

    def _mask_path_for(self, image_path: str) -> str | None:
        """
        Return the first existing mask path for this image basename, or None if none exist.
        We try a few common patterns emitted by builders.
        """
        base = os.path.splitext(os.path.basename(image_path))[0]
        mdir = os.path.join(self.cfg.DATASET_PATH, self.cfg.MASKS_DIR)
        candidates = [
            os.path.join(mdir, base + "_GT.png"),
            os.path.join(mdir, base + ".png"),
            os.path.join(mdir, base + "_mask.png"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    # ---- core override -----------------------------------------------------------

    def read_contents(self):
        """
        Build pos/neg samples directly from split paths.
        Uses KSDD2/Dataset utilities for reading, resizing and tensorization.
        """
        import os
        import numpy as np

        pos_samples, neg_samples = [], []

        paths, seg_set = self._split_paths()

        # ---- size handling: accept int or (W,H) from cfg/parent ----
        iw = getattr(self.cfg, "INPUT_WIDTH", None)
        ih = getattr(self.cfg, "INPUT_HEIGHT", None)

        if iw is not None and ih is not None:
            size_wh = (int(iw), int(ih))              # (W, H)
        else:
            s = getattr(self, "image_size", 512)
            if isinstance(s, (tuple, list)):
                size_wh = (int(s[0]), int(s[1]))
            else:
                s = int(s)
                size_wh = (s, s)

        grayscale = bool(self.cfg.INPUT_CHANNELS == 1)

        for img_path in paths:
            # Read image from the actual path in the split file
            image = self.read_img_resize(img_path, grayscale, size_wh)

            # Derive a mask path if it exists
            mpath = self._mask_path_for(img_path)

            # read_label_resize in this repo takes an **int** size; use max(W,H)
            label_size = max(size_wh[0], size_wh[1])

            if mpath is not None:
                seg_mask, positive = self.read_label_resize(mpath, size_wh, self.cfg.DILATE)
            else:
                seg_mask = np.zeros((size_wh[1], size_wh[0]), dtype=np.uint8)  # (H, W)
                positive = False

            is_segmented = (img_path in seg_set) if str(self.kind).upper() == "TRAIN" else False
            part = os.path.splitext(os.path.basename(img_path))[0]

            if positive:
                image_t = self.to_tensor(image)
                seg_loss_mask = self.distance_transform(
                    seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P
                )
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                seg_mask_t = self.to_tensor(self.downsize(seg_mask))
                pos_samples.append((image_t, seg_mask_t, seg_loss_mask, is_segmented, img_path, mpath, part))
            else:
                image_t = self.to_tensor(image)
                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                seg_mask_t = self.to_tensor(self.downsize(seg_mask))
                neg_samples.append((image_t, seg_mask_t, seg_loss_mask, True, img_path, mpath, part))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if str(self.kind).upper() == "TRAIN" else (len(pos_samples) + len(neg_samples))

        self.init_extra()

