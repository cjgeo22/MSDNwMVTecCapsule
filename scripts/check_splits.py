#!/usr/bin/env python3
import os, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True, help="Dataset root (the --dst you used for build_capsule_dataset.py)")
    args = ap.parse_args()

    ok = True
    for sub in ["KSDD2_like", "MVTecCapsule"]:
        base = os.path.join(args.dst, "splits", sub)
        for name in ["train.txt", "segmented_train.txt", "test.txt"]:
            p = os.path.join(base, name)
            if not os.path.exists(p):
                print("[ERR]", p, "missing")
                ok = False
                continue
            with open(p) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            for ln in lines[:5]:  # spot check some paths
                if not os.path.exists(ln):
                    print("[ERR]", sub, name, "path not found:", ln)
                    ok = False

        # verify masks for segmented_train
        seg = os.path.join(base, "segmented_train.txt")
        if os.path.exists(seg):
            with open(seg) as f:
                for img_path in [ln.strip() for ln in f if ln.strip()]:
                    mask_path = os.path.join(args.dst, "masks", os.path.basename(img_path).replace(".png", "_mask.png"))
                    if not os.path.exists(mask_path):
                        print("[ERR] mask missing for", img_path)
                        ok = False

    print("OK" if ok else "Found issues")

if __name__ == "__main__":
    main()
