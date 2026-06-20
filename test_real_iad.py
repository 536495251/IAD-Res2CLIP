"""
Real-IAD inference script.

Builds a memory bank from the training set, then performs 5-view inference
on a test set. Supports both training-free and finetune modes.

Usage:
    # Training-free inference
    python test_real_iad.py --test_path ./test_set --output ./output

    # Finetune inference (with trained adapters)
    python test_real_iad.py --test_path ./test_set --output ./output \
        --checkpoint ./checkpoints/iad/checkpoint_ep20.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
import zipfile

from models.clip import load_clean_clip, TextFeatureBank
from models.adapter import ResidualAdapter, MultiScaleConvAdapter
from models.knn import get_visual_match
from data.real_iad_dataset import RealIADDataset, SEEN_50_CATEGORIES
from utils.utils import get_transform
# Gaussian kernel for map smoothing (inline to avoid adeval dependency)
def get_gaussian_kernel(kernel_size=5, sigma=4, channels=1):
    import math
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                       kernel_size=kernel_size, groups=channels,
                                       bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_adapters(args, device):
    """Load fine-tuned adapters from checkpoint."""
    VIS_LAYERS  = args.visual_features_list
    RES_LAYERS  = args.res_features_list
    TEXT_LAYERS = args.text_features_list

    vis_adapters = nn.ModuleDict({str(l): MultiScaleConvAdapter() for l in VIS_LAYERS}).to(device)
    res_vis_local_adapters = nn.ModuleDict({str(l): MultiScaleConvAdapter() for l in RES_LAYERS}).to(device)
    res_vis_global_adapter = ResidualAdapter().to(device)
    res_text_adapter = ResidualAdapter().to(device)
    text_adapter = ResidualAdapter().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vis_adapters.load_state_dict(ckpt['vis_adapters'])
    res_vis_local_adapters.load_state_dict(ckpt['res_vis_local_adapters'])
    if 'res_vis_global_adapter' in ckpt:
        res_vis_global_adapter.load_state_dict(ckpt['res_vis_global_adapter'])
    if 'res_text_adapter' in ckpt:
        res_text_adapter.load_state_dict(ckpt['res_text_adapter'])
    if 'text_adapter' in ckpt:
        text_adapter.load_state_dict(ckpt['text_adapter'])

    for m in [vis_adapters, res_vis_local_adapters, res_vis_global_adapter,
              res_text_adapter, text_adapter]:
        m.eval()

    print(f"Checkpoint loaded from {args.checkpoint}")
    return vis_adapters, res_vis_local_adapters, res_vis_global_adapter, res_text_adapter, text_adapter


def build_memory_bank(model, args, preprocess, device, all_layers):
    """Build visual memory bank from the Real-IAD training set.

    Uses k_shot sample × all 5 views per class.
    Default k_shot=1 → 50 cls × 5 views = 250 reference images.
    This ensures query patches can find same-view matches.
    """
    train_dataset = RealIADDataset(
        root=args.train_path,
        transform=preprocess,
        target_transform=None,
        split='train',
        view_mode='single',
        view_index=None,  # iterate all 5 views
    )

    memory_bank = {}
    class_count = {}
    print("Building visual memory bank (all 5 views)...")
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(train_dataset)), desc="Memory bank"):
            items = train_dataset[idx]
            cls_name = items['cls_name']

            # Per class: keep k_shot samples × 5 views
            if cls_name not in class_count:
                class_count[cls_name] = 0
            if class_count[cls_name] >= args.k_shot * 5:
                continue
            class_count[cls_name] += 1

            image = items['img'].unsqueeze(0).to(device)
            image_features, patch_features = model.encode_image(
                image, all_layers, DPAM_layer=20)

            if cls_name not in memory_bank:
                memory_bank[cls_name] = {
                    'patch': [[] for _ in range(len(all_layers))],
                    'global': [],
                }

            feat_g = image_features / image_features.norm(dim=-1, keepdim=True)
            memory_bank[cls_name]['global'].append(feat_g.cpu())

            for i, feat in enumerate(patch_features):
                feat = feat[:, 1:, :]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                memory_bank[cls_name]['patch'][i].append(feat.cpu())

    # Stack along the shot dimension
    for cls_name in memory_bank:
        for i in range(len(memory_bank[cls_name]['patch'])):
            memory_bank[cls_name]['patch'][i] = torch.cat(
                memory_bank[cls_name]['patch'][i], dim=1)

        memory_bank[cls_name]['global'] = torch.cat(
            memory_bank[cls_name]['global'], dim=0).unsqueeze(0)

    total_refs = sum(m['global'].shape[1] for m in memory_bank.values())
    print(f"Memory bank built: {len(memory_bank)} categories, {total_refs} reference images")
    return memory_bank


def infer_one_view(model, text_bank, image, cls_name, memory_bank,
                   all_layers, text_layers, vis_layers, res_layers,
                   gaussian_kernel, img_size, args, adapters=None):
    """Run Res2CLIP inference on a single view image.

    Args:
        model:        frozen CLIP model
        text_bank:    TextFeatureBank
        image:        [1, 3, H, W] preprocessed image
        cls_name:     category name string
        memory_bank:  build_memory_bank() output
        all_layers:   sorted set of feature layers
        text_layers, vis_layers, res_layers: layer lists
        gaussian_kernel: for smoothing
        img_size:     output image size for anomaly maps
        args:         CLI args
        adapters:     tuple of 5 adapters (or None for training-free)

    Returns:
        anomaly_map:  [1, 1, img_size, img_size] smoothed map
        anomaly_score: float
        img_score_map: [1, 1, img_size, img_size] for pixel-level
    """
    B = 1
    with torch.no_grad():
        image_features, patch_features = model.encode_image(
            image, all_layers, DPAM_layer=20)

        # Text residual
        _, _, _, _, R_t = text_bank.get_features()
        if adapters is not None:
            *_, res_text_adapter, text_adapter = adapters
            R_t_res = res_text_adapter(R_t)
            R_t_res = R_t_res / R_t_res.norm(dim=-1, keepdim=True)
            R_t_text = text_adapter(R_t)
            R_t_text = R_t_text / R_t_text.norm(dim=-1, keepdim=True)
        else:
            R_t_res = R_t
            R_t_text = R_t

        ref_patch_features = [rpf.to(image.device) for rpf in memory_bank[cls_name]['patch']]

        maps_text, maps_vis, maps_res = [], [], []

        for i, layer_num in enumerate(all_layers):
            l_key = str(layer_num)
            feat = patch_features[i][:, 1:, :]
            feat = feat / feat.norm(dim=-1, keepdim=True)
            H_feat = int(feat.shape[1] ** 0.5)

            # Text branch
            if layer_num in text_layers:
                score_text = (feat @ R_t_text.unsqueeze(-1)).squeeze(-1)
                maps_text.append(score_text.view(B, 1, H_feat, H_feat))

            # KNN match → visual residual
            ref_feat = ref_patch_features[i]
            # Memory bank has 5 views per sample → k_shot × 5
            mem_k_shot = args.k_shot * 5
            _, matched_ref = get_visual_match(
                feat=feat, ref_feat=ref_feat,
                strategy=args.match_strategy,
                spatial_penalty=args.spatial_penalty,
                k_shot=mem_k_shot, device=image.device,
                use_sparse=args.use_sparse,
            )
            R_v = feat - matched_ref

            # Visual branch
            if layer_num in vis_layers:
                if adapters is not None:
                    vis_adapters, *_ = adapters
                    score_vis = torch.norm(vis_adapters[l_key](R_v), p=2, dim=-1)
                else:
                    score_vis = torch.norm(R_v, p=2, dim=-1)
                maps_vis.append(score_vis.view(B, 1, H_feat, H_feat))

            # Residual branch
            if layer_num in res_layers:
                if adapters is not None:
                    _, res_vis_local_adapters, *_ = adapters
                    proj = torch.clamp(
                        (res_vis_local_adapters[l_key](R_v) @ R_t_res.unsqueeze(-1)
                        ).squeeze(-1), min=0)
                else:
                    proj = torch.clamp(
                        (R_v @ R_t_res.unsqueeze(-1)).squeeze(-1), min=0)
                maps_res.append(proj.view(B, 1, H_feat, H_feat))

        # Aggregate and upsample maps
        def avg_upsample(maps):
            if not maps:
                return torch.zeros(1, 1, img_size, img_size, device=image.device)
            m = torch.stack(maps, dim=0).mean(dim=0)
            m = F.interpolate(m, size=(img_size, img_size), mode='bilinear')
            return gaussian_kernel(m)

        M_text = avg_upsample(maps_text)
        M_vis  = avg_upsample(maps_vis)
        M_res  = avg_upsample(maps_res)

        # Pixel-level fusion
        if adapters is not None:
            M = M_text * 0.1 + M_vis + M_res * 0.1
        else:
            M = M_text * 0.1 + M_vis + M_res
        M = F.interpolate(M, size=(img_size, img_size), mode='bilinear')
        M = gaussian_kernel(M)

        # Image-level score
        def top1pct(m):
            flat = m.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            return torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(dim=1)

        feat_global = image_features / image_features.norm(dim=-1, keepdim=True)

        # s_text_cls
        s_text_cls = (feat_global.unsqueeze(1) @ R_t_text.unsqueeze(-1)
                     ).squeeze(-1).squeeze(-1).unsqueeze(1)

        # Global visual residual
        ref_global = memory_bank[cls_name]['global'].to(image.device)
        ref_g_mean = ref_global.mean(dim=1, keepdim=True)
        ref_g_mean = ref_g_mean / ref_g_mean.norm(dim=-1, keepdim=True)
        R_v_cls = feat_global.unsqueeze(1) - ref_g_mean
        if adapters is not None:
            *_, res_vis_global_adapter, _, _ = adapters
            R_v_cls_res = res_vis_global_adapter(R_v_cls)
            s_res_cls = torch.clamp(
                (R_v_cls_res @ R_t_res.unsqueeze(-1)).squeeze(-1).squeeze(-1),
                min=0).unsqueeze(1)
        else:
            s_res_cls = torch.clamp(
                (R_v_cls @ R_t.unsqueeze(-1)).squeeze(-1).squeeze(-1),
                min=0).unsqueeze(1)

        s_text = 0.5 * top1pct(M_text).unsqueeze(1) + 0.5 * s_text_cls
        s_vis  = top1pct(M_vis).unsqueeze(1)
        s_res  = 0.5 * top1pct(M_res).unsqueeze(1) + 0.5 * s_res_cls

        if adapters is not None:
            s = s_text + s_vis + s_res * 0.1
        else:
            s = s_text + s_vis + s_res

        return M, s.item()


def infer_sample(model, text_bank, sample_dir, cls_name, memory_bank,
                 all_layers, text_layers, vis_layers, res_layers,
                 gaussian_kernel, img_size, args, adapters=None, preprocess=None):
    """Run 5-view inference on one sample.

    Returns aggregated sample score and per-view anomaly maps.
    """
    scores = []
    maps = []

    for view_idx in range(5):
        img_path = os.path.join(sample_dir, f"{view_idx}.png")
        if not os.path.exists(img_path):
            continue

        # Load and preprocess
        img = Image.open(img_path).convert('RGB')
        image = preprocess(img).unsqueeze(0).to(args.device)

        # Infer
        anom_map, score = infer_one_view(
            model, text_bank, image, cls_name, memory_bank,
            all_layers, text_layers, vis_layers, res_layers,
            gaussian_kernel, img_size, args, adapters)
        scores.append(score)
        maps.append(anom_map)

    # Aggregate score: Max fusion (default)
    sample_score = max(scores)

    return sample_score, maps, scores


def generate_submission(results, output_path, mask_size=448):
    """Generate submission.csv + predicted_masks/ and zip them."""
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "submission.csv")
    mask_dir = os.path.join(output_path, "predicted_masks")

    # 1. Write submission.csv
    rows = []
    for r in results:
        rows.append({
            "group_folder": r["group_folder"],
            "anomaly_score": f"{r['anomaly_score']:.6f}",
        })

    with open(csv_path, "w") as f:
        f.write("group_folder,anomaly_score\n")
        for row in rows:
            f.write(f"{row['group_folder']},{row['anomaly_score']}\n")

    # 2. Write predicted_masks/
    for r in results:
        cat, sample = r["group_folder"].split("/")
        sample_mask_dir = os.path.join(mask_dir, cat, sample)
        os.makedirs(sample_mask_dir, exist_ok=True)

        for view_idx, anom_map in enumerate(r["per_view_masks"]):
            # Upsample to 448×448
            mask_448 = F.interpolate(anom_map, size=(mask_size, mask_size),
                                     mode='bilinear', align_corners=False)
            # Normalize to [0, 1]
            mask_min = mask_448.min()
            mask_448 = (mask_448 - mask_min) / (mask_448.max() - mask_min + 1e-8)
            # Quantize to uint8
            mask_np = (mask_448.squeeze().cpu().numpy() * 255).astype(np.uint8)
            # Save
            mask_path = os.path.join(sample_mask_dir, f"{view_idx}_mask.png")
            Image.fromarray(mask_np, mode='L').save(mask_path)

    # 3. Zip
    zip_path = os.path.join(output_path, "submission.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")
        for root, dirs, files in os.walk(mask_dir):
            for f in files:
                file_path = os.path.join(root, f)
                arcname = os.path.relpath(file_path, output_path)
                zf.write(file_path, arcname=arcname)

    print(f"  submission.csv: {csv_path}")
    print(f"  masks:          {mask_dir}/")
    print(f"  zip:            {zip_path}")


def main(args):
    setup_seed(args.seed)
    device = args.device
    img_size = args.image_size
    output_path = os.path.abspath(args.output)
    os.makedirs(output_path, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────
    print("Loading CLIP ViT-L/14@336px...")
    model, _ = load_clean_clip('ViT-L/14@336px', device=device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ── Feature layers ─────────────────────────────────────────────
    TEXT_LAYERS = args.text_features_list
    VIS_LAYERS  = args.visual_features_list
    RES_LAYERS  = args.res_features_list
    ALL_LAYERS  = sorted(set(TEXT_LAYERS + VIS_LAYERS + RES_LAYERS))

    # ── Data transforms ────────────────────────────────────────────
    args_t = argparse.Namespace(image_size=img_size)
    preprocess, target_transform = get_transform(args_t)

    # ── Text feature bank ──────────────────────────────────────────
    print("Initializing text feature bank...")
    text_bank = TextFeatureBank(model, device)

    # ── Memory bank ────────────────────────────────────────────────
    memory_bank = build_memory_bank(model, args, preprocess, device, ALL_LAYERS)

    # ── Load adapters (optional) ───────────────────────────────────
    adapters = None
    if args.checkpoint:
        adapters = load_adapters(args, device)
        mode_str = "finetune"
    else:
        mode_str = "training-free"
    print(f"Mode: {mode_str}")

    # ── Gaussian kernel ────────────────────────────────────────────
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    # ── Scan test set ──────────────────────────────────────────────
    print(f"\nScanning test set: {args.test_path}")
    test_categories = sorted(os.listdir(args.test_path))
    # Filter to known categories (or use all)
    test_samples = []
    for cat in test_categories:
        cat_path = os.path.join(args.test_path, cat)
        if not os.path.isdir(cat_path):
            continue
        samples = sorted(
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d))
        )
        for s in samples:
            sample_dir = os.path.join(cat_path, s)
            # Verify at least view 0 exists
            if os.path.isfile(os.path.join(sample_dir, "0.png")):
                test_samples.append((cat, sample_dir))

    print(f"Found {len(test_samples)} samples in {len(test_categories)} categories")

    # ── Inference ──────────────────────────────────────────────────
    results = []
    all_masks_zip = {}  # for zip

    for cls_name, sample_dir in tqdm(test_samples, desc="Inferring"):
        # Get relative group_folder path
        rel_path = os.path.relpath(sample_dir, args.test_path)

        # Run 5-view inference
        sample_score, maps, view_scores = infer_sample(
            model, text_bank, sample_dir, cls_name, memory_bank,
            ALL_LAYERS, TEXT_LAYERS, VIS_LAYERS, RES_LAYERS,
            gaussian_kernel, img_size, args, adapters, preprocess)

        results.append({
            "group_folder": rel_path,
            "anomaly_score": sample_score,
            "per_view_masks": maps,
        })

    # ── Generate submission ────────────────────────────────────────
    print(f"\nGenerating submission files...")
    generate_submission(results, output_path, mask_size=args.mask_size)

    # ── Summary ────────────────────────────────────────────────────
    scores = [r["anomaly_score"] for r in results]
    print(f"\n{'='*60}")
    print(f"Results summary ({mode_str})")
    print(f"{'='*60}")
    print(f"  Samples:        {len(results)}")
    print(f"  Score range:    [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Score mean:     {np.mean(scores):.4f}")
    print(f"  Score median:   {np.median(scores):.4f}")

    # Quick distribution check
    high = sum(1 for s in scores if s > 0.5)
    mid  = sum(1 for s in scores if 0.1 < s <= 0.5)
    low  = sum(1 for s in scores if s <= 0.1)
    print(f"  Distribution:   low≤0.1: {low}  mid: {mid}  high>0.5: {high}")
    print(f"{'='*60}\n")

    print(f"Output saved to: {output_path}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Res2CLIP Real-IAD Inference")

    # Paths
    parser.add_argument('--train_path', type=str, default='./dataset',
                        help='Real-IAD training set (for memory bank)')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Test set path (e.g. ./dataset/Test_A)')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for submission files')

    # Mode
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to finetuned checkpoint (None = training-free)')

    # Feature layers
    parser.add_argument('--text_features_list',   type=int, nargs='+', default=[24])
    parser.add_argument('--visual_features_list', type=int, nargs='+', default=[6, 12, 18, 24])
    parser.add_argument('--res_features_list',    type=int, nargs='+', default=[24])

    # Matching
    parser.add_argument('--match_strategy',  type=str,  default='sparse_radial_knn')
    parser.add_argument('--spatial_penalty', type=float, default=0.01)
    parser.add_argument('--use_sparse',      action='store_true')
    parser.add_argument('--k_shot',          type=int,   default=1)

    # Image settings
    parser.add_argument('--image_size', type=int, default=518,
                        help='Input image size for CLIP')
    parser.add_argument('--mask_size', type=int, default=448,
                        help='Output mask size for submission')

    # Misc
    parser.add_argument('--device', type=str, default='cuda',
                        help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=111)

    args = parser.parse_args()

    print("=" * 60)
    print("Res2CLIP — Real-IAD Inference")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    main(args)
