"""
Generate metadata JSON for Real-IAD dataset.

Scans the Train (and optionally Test_A / Test_B) directories and produces
a single `real_iad_meta.json` that can be loaded by RealIADDataset.

Output format:
{
    "train": {
        "3_adapter": {
            "normal": [
                "Train/3_adapter/S0001",
                "Train/3_adapter/S0002",
                ...
            ]
        },
        ...
    },
    "test_a": { ... },   (optional)
    "test_b": { ... }    (optional)
}
"""

import json
import os
import argparse
from glob import glob


def scan_split(root, split_name="Train"):
    """Scan one split (Train / Test_A / Test_B) and return category->samples dict."""
    split_path = os.path.join(root, split_name)
    if not os.path.isdir(split_path):
        print(f"  [SKIP] {split_name} not found at {split_path}")
        return None

    result = {}
    categories = sorted(os.listdir(split_path))
    for cat in categories:
        cat_path = os.path.join(split_path, cat)
        if not os.path.isdir(cat_path):
            continue

        # Find all sample folders (S0001, S0002, ...)
        samples = sorted(
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d))
        )
        sample_paths = []
        for s in samples:
            sample_dir = os.path.join(split_name, cat, s)
            # Verify at least view 0 exists
            if os.path.isfile(os.path.join(root, sample_dir, "0.png")):
                sample_paths.append(sample_dir)

        if sample_paths:
            result[cat] = {"normal": sample_paths}

    return result


def generate_meta(root, splits):
    """Generate full metadata dict for given splits."""
    meta = {}
    for split_name in splits:
        print(f"Scanning {split_name}...")
        data = scan_split(root, split_name)
        if data is not None:
            key = split_name.lower().replace(" ", "_")
            meta[key] = data
            n_cats = len(data)
            n_samples = sum(len(v["normal"]) for v in data.values())
            print(f"  → {n_cats} categories, {n_samples} samples")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Real-IAD metadata")
    parser.add_argument("--root", type=str, default="./dataset",
                        help="Root directory containing Train/ (and optional Test_A/, Test_B/)")
    parser.add_argument("--output", type=str, default="./data/real_iad_meta.json",
                        help="Output JSON path")
    parser.add_argument("--splits", type=str, nargs="+", default=["Train"],
                        help="Splits to scan (default: Train)")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    print(f"Root: {root}")
    print(f"Splits: {args.splits}")

    meta = generate_meta(root, args.splits)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {args.output}")
    total_samples = sum(
        len(v["normal"])
        for split_data in meta.values()
        for v in split_data.values()
    )
    total_cats = sum(len(split_data) for split_data in meta.values())
    print(f"Total: {total_cats} categories, {total_samples} samples across {list(meta.keys())}")
