#!/usr/bin/env python3
# Create KSDD2-like and "proper" MVTecCapsule splits for the Mixed-SegDec-Net repo.
# - Copies selected images from MVTec AD capsule into a flat `images/` dir
# - Copies available masks into `masks/`
# - Writes split files:
#     * splits/KSDD2_like/{train.txt, segmented_train.txt, test.txt}
#     * splits/MVTecCapsule/{train.txt, segmented_train.txt, test.txt}
# You can adjust how many masked positives go into training.

import os, shutil, random, argparse, pathlib

def list_pngs(root):
    return sorted([str(p) for p in pathlib.Path(root).rglob("*.png")])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to mvtec_ad/capsule")
    ap.add_argument("--dst", required=True, help="Output dataset root (e.g., /data/MVTEC_CAPSULE)")
    ap.add_argument("--num_masked_train", type=int, default=40, help="How many anomalous (with masks) to move into train")
    ap.add_argument("--train_normal_frac", type=float, default=1.0, help="Fraction of train/good to use as normal training")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    SRC = args.src
    DST = args.dst

    images_dir = os.path.join(DST, "images")
    masks_dir = os.path.join(DST, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    splits_ks = os.path.join(DST, "splits", "KSDD2_like")
    splits_mv = os.path.join(DST, "splits", "MVTecCapsule")
    os.makedirs(splits_ks, exist_ok=True)
    os.makedirs(splits_mv, exist_ok=True)

    # Collect source
    train_good = list_pngs(os.path.join(SRC, "train", "good"))
    test_good = list_pngs(os.path.join(SRC, "test", "good"))
    test_bad = []
    test_dir = os.path.join(SRC, "test")
    for d in pathlib.Path(test_dir).iterdir():
        if d.is_dir() and d.name != "good":
            test_bad += list_pngs(str(d))
    test_bad = sorted(test_bad)

    def mask_for(img_path):
        rel = pathlib.Path(img_path).relative_to(os.path.join(SRC, "test"))
        defect_dir = rel.parts[0]         # e.g., 'broken_large'
        stem = pathlib.Path(rel.parts[-1]).stem
        m_rel = os.path.join(defect_dir, f"{stem}_mask.png")
        return os.path.join(SRC, "ground_truth", m_rel)

    # Select masked train positives
    chosen_train_pos = random.sample(test_bad, k=min(args.num_masked_train, len(test_bad)))
    chosen_set = set(chosen_train_pos)

    # Split holders (we will use identical content for both 'KSDD2_like' and 'MVTecCapsule')
    train_lines = []
    seg_train_lines = []
    test_lines = []

    # Normals to train
    k = int(len(train_good) * args.train_normal_frac)
    for p in train_good[:k]:
        dst = os.path.join(images_dir, "train_good__" + os.path.basename(p))
        shutil.copy2(p, dst)
        train_lines.append(os.path.abspath(dst))

    # Masked positives into train
    for p in chosen_train_pos:
        m = mask_for(p)
        if not os.path.exists(m):
            # Shouldn't happen in MVTec; skip if missing
            continue
        # encode source subdirs into filename to keep class hints
        sub = pathlib.Path(*pathlib.Path(p).parts[-2:]).as_posix().replace("/", "__")
        img_dst = os.path.join(images_dir, "train_bad__" + sub)
        msk_dst = os.path.join(masks_dir, os.path.basename(img_dst).replace(".png", "_mask.png"))
        shutil.copy2(p, img_dst)
        shutil.copy2(m, msk_dst)
        abspath = os.path.abspath(img_dst)
        train_lines.append(abspath)
        seg_train_lines.append(abspath)

    # Remaining test set
    for p in test_good:
        img_dst = os.path.join(images_dir, "test_good__" + os.path.basename(p))
        shutil.copy2(p, img_dst)
        test_lines.append(os.path.abspath(img_dst))

    for p in test_bad:
        if p in chosen_set:
            continue
        sub = pathlib.Path(*pathlib.Path(p).parts[-2:]).as_posix().replace("/", "__")
        img_dst = os.path.join(images_dir, "test_bad__" + sub)
        shutil.copy2(p, img_dst)
        test_lines.append(os.path.abspath(img_dst))
        m = mask_for(p)
        if os.path.exists(m):
            msk_dst = os.path.join(masks_dir, os.path.basename(img_dst).replace(".png", "_mask.png"))
            shutil.copy2(m, msk_dst)

    # Write both sets of splits (same content; different directory for clarity)
    def write_splits(splits_dir):
        with open(os.path.join(splits_dir, "train.txt"), "w") as f:
            f.write("\n".join(train_lines) + "\n")
        with open(os.path.join(splits_dir, "segmented_train.txt"), "w") as f:
            f.write("\n".join(seg_train_lines) + "\n")
        with open(os.path.join(splits_dir, "test.txt"), "w") as f:
            f.write("\n".join(test_lines) + "\n")

    write_splits(splits_ks)
    write_splits(splits_mv)

    print("Done.")
    print("DST:", DST)
    print("Train:", len(train_lines), " Segmented train:", len(seg_train_lines), " Test:", len(test_lines))

if __name__ == "__main__":
    main()
