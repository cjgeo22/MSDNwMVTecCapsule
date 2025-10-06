# data/input_ksdd2.py
# KSDD2 dataset loader (unchanged logic) + robust image reader.
# The only functional addition is _imread_strict() and an override of read_img_resize()
# to make image decoding resilient to odd PNGs that OpenCV sometimes fails to load.

import os
import pickle
import numpy as np
import cv2
from PIL import Image  # fallback decoder

from data.dataset import Dataset
from config import Config


def _imread_strict(path: str, grayscale: bool):
    """
    Try cv2 first; if it returns None, fall back to PIL.
    Return shape like cv2.imread:
      - grayscale=True  -> 2D (H, W) uint8
      - grayscale=False -> 3D (H, W, 3) uint8 in **BGR** order
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)

    if img is None:
        # PIL fallback (handles some PNGs that trip cv2)
        try:
            pil = Image.open(path)
            pil = pil.convert("L" if grayscale else "RGB")
            img = np.array(pil)

            if not grayscale:
                # Convert RGB (PIL) -> BGR (cv2 convention) to preserve original pipeline semantics
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise FileNotFoundError(f"Failed to read image: {path} ({e})")

    # Ensure dtype is uint8 (expected by downstream cv2 ops)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def read_split(num_segmented: int, kind: str):
    """
    Original split reader: expects a pickle with two lists of (part_id, is_segmented_bool).
    """
    fn = f"KSDD2/split_{num_segmented}.pyb"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == "TRAIN":
            return train_samples
        elif kind == "TEST":
            return test_samples
        else:
            raise Exception("Unknown")


class KSDD2Dataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(KSDD2Dataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    # --- New: override to use the strict reader --------------------------------
    def read_img_resize(self, image_path: str, grayscale: bool, image_size):
        """
        image_size can be an int (square) or a (W,H) tuple/list.
        """
        img = _imread_strict(image_path, grayscale)

        if isinstance(image_size, (tuple, list)):
            w, h = int(image_size[0]), int(image_size[1])
            dsize = (w, h)                   # cv2 takes (W, H)
        else:
            s = int(image_size)
            dsize = (s, s)

        img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        return img

    # ---------------------------------------------------------------------------

    def read_contents(self):
        """
        Original KSDD2 loader logic:
          - read (part_id, is_segmented_bool)
          - compose paths: <DATASET_PATH>/<kind>/<part>.png and <part>_GT.png
          - read/resize, build pos/neg lists
        """
        pos_samples, neg_samples = [], []

        data_points = read_split(self.cfg.NUM_SEGMENTED, self.kind)

        for part, is_segmented in data_points:
            image_path = os.path.join(self.path, self.kind.lower(), f"{part}.png")
            seg_mask_path = os.path.join(self.path, self.kind.lower(), f"{part}_GT.png")

            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

            if positive:
                image = self.to_tensor(image)
                seg_loss_mask = self.distance_transform(
                    seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P
                )
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, part))
            else:
                image = self.to_tensor(image)
                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        # TRAIN uses each positive twice (seg + dec); TEST uses all once
        self.len = 2 * len(pos_samples) if self.kind in ["TRAIN"] else len(pos_samples) + len(neg_samples)

        self.init_extra()
