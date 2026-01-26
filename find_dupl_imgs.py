#!/usr/bin/env python3
import argparse
import hashlib
import os
from pathlib import Path

# Default image formats supported by matplotlib
MATPLOTLIB_FORMATS = {
    "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "svg", "eps", "pdf"
}

def hash_file(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def collect_images(folder, exts):
    files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            if not exts or name.lower().split(".")[-1] in exts:
                files.append(Path(root) / name)
    return files

def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder_a")
    p.add_argument("folder_b")
    p.add_argument(
        "--ext",
        nargs="*",
        help="Filter by one or more extensions (e.g. --ext png jpg). If omitted, all matplotlib formats are used."
    )
    args = p.parse_args()

    exts = set([e.lower().lstrip(".") for e in args.ext]) if args.ext else MATPLOTLIB_FORMATS

    images_a = collect_images(args.folder_a, exts)
    images_b = collect_images(args.folder_b, exts)

    hashes_b = {}
    for img in images_b:
        h = hash_file(img)
        hashes_b.setdefault(h, []).append(img)

    for img_a in images_a:
        h = hash_file(img_a)
        if h in hashes_b:
            print("Duplicate:", img_a)
            for match in hashes_b[h]:
                print("  matches:", match)

if __name__ == "__main__":
    main()