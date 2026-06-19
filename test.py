import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import warnings

from models.clip import load_clean_clip, TextFeatureBank
from models.adapter import ResidualAdapter, MultiScaleConvAdapter
from models.knn import get_visual_match
from data.dataset import Dataset
from utils.utils import get_transform
from utils.metrics import ader_evaluator, get_gaussian_kernel
from utils.logger import get_logger

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _encode_ref_image(img_path, preprocess, model, all_features_list, device):
    '''Load one reference image from path and return (image_features, patch_features).'''
    img = Image.open(img_path).convert('RGB')
    image = preprocess(img).unsqueeze(0).to(device)
    return model.encode_image(image, all_features_list, DPAM_layer=20)


def build_visual_memory(model, args, preprocess, target_transform, device, all_features_list):
    '''Build K-shot visual memory bank.

    Reference image selection (priority order):
      1. --ref_json <path>  explicit JSON {cls: [img_path, ...]} — exact reproducibility
      2. auto-record        shuffle with --seed, save/reload from few_shot_records/ next to data dir
    '''
    memory_bank = {}

    if args.ref_json:
        # Fast path: paths are already known — load only the listed images directly.
        if not os.path.exists(args.ref_json):
            raise FileNotFoundError(f"--ref_json path not found: {args.ref_json}")
        print(f" > Loading few-shot record: {args.ref_json}")
        with open(args.ref_json, 'r') as f:
            ref_paths = json.load(f)  # {cls_name: [abs_img_path, ...]}

        model.eval()
        with torch.no_grad():
            for cls_name, paths in tqdm(ref_paths.items(), desc="Building memory bank"):
                for img_path in paths[:args.k_shot]:
                    image_features, patch_features = _encode_ref_image(
                        img_path, preprocess, model, all_features_list, device)

                    if cls_name not in memory_bank:
                        memory_bank[cls_name] = {
                            'patch': [[] for _ in range(len(patch_features))],
                            'global': [],
                        }
                    feat_g = image_features / image_features.norm(dim=-1, keepdim=True)
                    memory_bank[cls_name]['global'].append(feat_g)
                    for i, feat in enumerate(patch_features):
                        feat = feat[:, 1:, :]
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        memory_bank[cls_name]['patch'][i].append(feat)

    else:
        # Slow path: no JSON provided — scan the training set, auto-select and record.
        train_data = Dataset(root=args.data_path, transform=preprocess,
                             target_transform=target_transform,
                             dataset_name=args.dataset, mode='train')
        record_dir = os.path.join(os.path.dirname(os.path.abspath(args.data_path)), "few_shot_records")
        os.makedirs(record_dir, exist_ok=True)
        record_filename = f"{args.dataset}_{args.k_shot}shot_seed{args.seed}_shuffle{args.shuffle}.json"
        record_path = os.path.join(record_dir, record_filename)

        selected_paths = {cls_name: [] for cls_name in train_data.obj_list}
        class_shot_counts = {cls_name: 0 for cls_name in train_data.obj_list}
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=args.shuffle)

        print(f" > No record found. Selecting images (seed={args.seed}, shuffle={args.shuffle}) → {record_path}")
        model.eval()
        with torch.no_grad():
            for items in tqdm(train_dataloader, desc="Building memory bank"):
                cls_name = items['cls_name'][0]
                if items['anomaly'][0] != 0:
                    continue
                if class_shot_counts[cls_name] >= args.k_shot:
                    continue

                image = items['img'].to(device)
                image_features, patch_features = model.encode_image(image, all_features_list, DPAM_layer=20)

                if cls_name not in memory_bank:
                    memory_bank[cls_name] = {
                        'patch': [[] for _ in range(len(patch_features))],
                        'global': [],
                    }
                feat_g = image_features / image_features.norm(dim=-1, keepdim=True)
                memory_bank[cls_name]['global'].append(feat_g)
                for i, feat in enumerate(patch_features):
                    feat = feat[:, 1:, :]
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    memory_bank[cls_name]['patch'][i].append(feat)

                class_shot_counts[cls_name] += 1
                selected_paths[cls_name].append(items['img_path'][0])

                if all(count == args.k_shot for count in class_shot_counts.values()):
                    break

        with open(record_path, 'w') as f:
            json.dump(selected_paths, f, indent=4)
        print(f" > Saved few-shot record to {record_path}")

    # Stack along the shot dimension
    for cls_name in memory_bank:
        for i in range(len(memory_bank[cls_name]['patch'])):
            memory_bank[cls_name]['patch'][i] = torch.cat(memory_bank[cls_name]['patch'][i], dim=1)
        memory_bank[cls_name]['global'] = torch.cat(memory_bank[cls_name]['global'], dim=0).unsqueeze(0)

    return memory_bank


def load_adapters(args, device):
    '''Load fine-tuned adapters from checkpoint.'''
    # Visual branch: {A_{v,l}} — one adapter per visual layer
    vis_adapters = nn.ModuleDict({str(l): MultiScaleConvAdapter() for l in args.visual_features_list}).to(device)
    # Residual branch: {A^res_{v,l}} — one adapter per residual layer
    res_vis_local_adapters = nn.ModuleDict({str(l): MultiScaleConvAdapter() for l in args.res_features_list}).to(device)
    # Residual branch: A^res_{v,cls} — global visual adapter
    res_vis_global_adapter = ResidualAdapter().to(device)
    # Residual branch: A^res_t — text residual adapter
    res_text_adapter = ResidualAdapter().to(device)
    # Text branch: A_t — text residual adapter
    text_adapter = ResidualAdapter().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    vis_adapters.load_state_dict(ckpt['vis_adapters'])
    res_vis_local_adapters.load_state_dict(ckpt['res_vis_local_adapters'])
    if 'res_vis_global_adapter' in ckpt:
        res_vis_global_adapter.load_state_dict(ckpt['res_vis_global_adapter'])
    if 'res_text_adapter' in ckpt:
        res_text_adapter.load_state_dict(ckpt['res_text_adapter'])
    if 'text_adapter' in ckpt:
        text_adapter.load_state_dict(ckpt['text_adapter'])

    for m in [vis_adapters, res_vis_local_adapters, res_vis_global_adapter, res_text_adapter, text_adapter]:
        m.eval()

    print("Checkpoint loaded successfully.")
    return vis_adapters, res_vis_local_adapters, res_vis_global_adapter, res_text_adapter, text_adapter


def test(args):
    setup_seed(args.seed)
    img_size = args.image_size
    save_path = os.path.join(args.save_path, args.mode)
    os.makedirs(save_path, exist_ok=True)

    logger = get_logger(save_path)
    logger.info(args)
    logger.info(f"Mode: {args.mode} | Match: {args.match_strategy}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = load_clean_clip("ViT-L/14@336px", device=device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if args.mode == 'finetune':
        vis_adapters, res_vis_local_adapters, res_vis_global_adapter, \
            res_text_adapter, text_adapter = load_adapters(args, device)

    TEXT_LAYERS = args.text_features_list
    VIS_LAYERS  = args.visual_features_list
    RES_LAYERS  = args.res_features_list
    ALL_LAYERS  = sorted(set(TEXT_LAYERS + VIS_LAYERS + RES_LAYERS))

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform,
                        dataset_name=args.dataset, mode='test')
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    logger.info("Initializing text feature bank...")
    text_bank = TextFeatureBank(model, device)
    logger.info("Initializing visual memory bank...")
    visual_memory = build_visual_memory(model, args, preprocess, target_transform, device, ALL_LAYERS)

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    gt_list_px, pr_list_px, gt_list_sp, pr_list_sp = [], [], [], []
    current_cls = None
    metric_results = []

    for items in tqdm(test_dataloader, desc="Testing"):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        label = items['anomaly']

        if cls_name[0] != current_cls:
            if current_cls is not None:
                gt_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
                pr_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
                gt_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
                pr_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_px, pr_sp, gt_px, gt_sp)
                result = {"Class": current_cls, "I-AUROC": auroc_sp, "I-AP": ap_sp, "I-F1": f1_sp,
                          "P-AUROC": auroc_px, "P-AP": ap_px, "P-F1": f1_px, "PRO": aupro_px}
                metric_results.append(result)
                logger.info(f" > {current_cls:<15} | I-AUC: {auroc_sp:.4f} | P-AUC: {auroc_px:.4f} | PRO: {aupro_px:.4f}")
                gt_list_px, pr_list_px, gt_list_sp, pr_list_sp = [], [], [], []
            current_cls = cls_name[0]

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, ALL_LAYERS, DPAM_layer=20)

            # ── Text residual preparation ─────────────────────────────────────────
            # R_t: raw text residual (= F̂^a_t − F̂^n_t, frozen, unit-norm)
            _, _, _, _, R_t = text_bank.get_features()
            if args.mode == 'finetune':
                # R̃^res_t = A^res_t(R_t): adapted text residual for residual branch
                R_t_res = res_text_adapter(R_t)
                R_t_res = R_t_res / R_t_res.norm(dim=-1, keepdim=True)
                # R̃_t = A_t(R_t): adapted text residual for text branch
                R_t_text = text_adapter(R_t)
                R_t_text = R_t_text / R_t_text.norm(dim=-1, keepdim=True)
            else:
                R_t_res  = R_t  # identity in training-free mode
                R_t_text = R_t

            ref_patch_features = visual_memory[cls_name[0]]['patch']

            # ── Local (pixel-level) score maps ────────────────────────────────────
            maps_text, maps_vis, maps_res = [], [], []

            for i, layer_num in enumerate(ALL_LAYERS):
                l_key = str(layer_num)
                feat = patch_features[i][:, 1:, :]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                B, L, D = feat.shape
                H_feat = int(L ** 0.5)

                # Text branch: ⟨F^q_{v,l}, R̃_t⟩
                if layer_num in TEXT_LAYERS:
                    score_text = (feat @ R_t_text.unsqueeze(-1)).squeeze(-1)
                    maps_text.append(score_text.view(B, 1, H_feat, H_feat))

                # KNN match → visual residual R_{v,l} = F̂^q_{v,l} − F̂_ref,l
                ref_feat = ref_patch_features[i]
                _, matched_ref = get_visual_match(
                    feat=feat, ref_feat=ref_feat,
                    strategy=args.match_strategy,
                    spatial_penalty=args.spatial_penalty,
                    k_shot=args.k_shot, device=device,
                    use_sparse=args.use_sparse,
                )
                R_v = feat - matched_ref

                # Visual branch: ‖R̃_{v,l}‖²
                if layer_num in VIS_LAYERS:
                    if args.mode == 'finetune':
                        score_vis = torch.norm(vis_adapters[l_key](R_v), p=2, dim=-1)
                    else:
                        score_vis = torch.norm(R_v, p=2, dim=-1)
                    maps_vis.append(score_vis.view(B, 1, H_feat, H_feat))

                # Residual branch: R̃^res_{v,l} · R̂^res_t 
                if layer_num in RES_LAYERS:
                    if args.mode == 'finetune':
                        proj = torch.clamp(
                            (res_vis_local_adapters[l_key](R_v) @ R_t_res.unsqueeze(-1)).squeeze(-1), min=0)
                    else:
                        proj = torch.clamp((R_v @ R_t_res.unsqueeze(-1)).squeeze(-1), min=0)
                    maps_res.append(proj.view(B, 1, H_feat, H_feat))

            # ── Aggregate and upsample maps ───────────────────────────────────────
            def avg_upsample(maps):
                m = torch.stack(maps, dim=0).mean(dim=0)
                m = F.interpolate(m, size=(img_size, img_size), mode='bilinear')
                return gaussian_kernel(m)          

            M_text = avg_upsample(maps_text)
            M_vis  = avg_upsample(maps_vis)
            M_res  = avg_upsample(maps_res)

            # ── Pixel-level fusion ──────────────
            if args.mode == 'finetune':
                M = M_text * 0.1 + M_vis + M_res * 0.1
            else:
                M = M_text * 0.1 + M_vis + M_res
            M = F.interpolate(M, size=(img_size, img_size), mode='bilinear')
            M = gaussian_kernel(M)
            gt_list_px.append(gt_mask)
            pr_list_px.append(M)

            # ── Image-level scores ────────────────────────────────
            def top1pct(m):
                flat = m.flatten(1)
                k = max(1, int(flat.shape[1] * 0.01))
                return torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(dim=1).unsqueeze(1)

            feat_global = image_features / image_features.norm(dim=-1, keepdim=True)

            # s_text_cls = ⟨f^q_v, R̃_t⟩
            s_text_cls = (feat_global.unsqueeze(1) @ R_t_text.unsqueeze(-1)).squeeze(-1).squeeze(-1).unsqueeze(1)

            # R_{v,cls} = f^q_v − f̄_ref  (raw global visual residual)
            ref_global = visual_memory[cls_name[0]]['global']
            ref_g_mean = ref_global.mean(dim=1, keepdim=True)
            ref_g_mean = ref_g_mean / ref_g_mean.norm(dim=-1, keepdim=True)
            R_v_cls = feat_global.unsqueeze(1) - ref_g_mean
            if args.mode == 'finetune':
                # R̃^res_{v,cls} = A^res_{v,cls}(R_{v,cls})
                R_v_cls_res = res_vis_global_adapter(R_v_cls)
                s_res_cls = torch.clamp(
                    (R_v_cls_res @ R_t_res.unsqueeze(-1)).squeeze(-1).squeeze(-1), min=0).unsqueeze(1)
            else:
                s_res_cls = torch.clamp(
                    (R_v_cls @ R_t.unsqueeze(-1)).squeeze(-1).squeeze(-1), min=0).unsqueeze(1)

            s_text = 0.5 * top1pct(M_text) + 0.5 * s_text_cls
            s_vis  = top1pct(M_vis)
            s_res  = 0.5 * top1pct(M_res) + 0.5 * s_res_cls

            if args.mode == 'finetune':
                s = s_text + s_vis + s_res * 0.1
            else:
                s = s_text + s_vis + s_res

            gt_list_sp.append(label)
            pr_list_sp.append(s.squeeze(1))

    # Last class
    if current_cls is not None:
        gt_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_px, pr_sp, gt_px, gt_sp)
        result = {"Class": current_cls, "I-AUROC": auroc_sp, "I-AP": ap_sp, "I-F1": f1_sp,
                  "P-AUROC": auroc_px, "P-AP": ap_px, "P-F1": f1_px, "PRO": aupro_px}
        metric_results.append(result)
        logger.info(f" > {current_cls:<15} | I-AUC: {auroc_sp:.4f} | P-AUC: {auroc_px:.4f} | PRO: {aupro_px:.4f}")

    df = pd.DataFrame(metric_results)
    mean_row = df.mean(numeric_only=True)
    mean_row["Class"] = "AVERAGE"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df = df[["Class", "I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "PRO"]]

    logger.info("\n" + "=" * 80)
    logger.info(f"RESULTS — {args.mode} | {args.dataset} | {args.k_shot}-shot | match={args.match_strategy}")
    logger.info("-" * 80)
    logger.info(df.to_string(index=False, float_format="%.4f"))
    logger.info("=" * 80 + "\n")

    csv_name = f"{args.dataset}_{args.mode}_{args.k_shot}shot_seed{args.seed}.csv"
    df.to_csv(os.path.join(save_path, csv_name), index=False)

    print("\n" + "=" * 80)
    print(df.to_string(index=False, float_format="%.4f"))
    print("=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Res2CLIP Evaluation")

    # Mode
    parser.add_argument('--mode', type=str, required=True, choices=['training-free', 'finetune'],
                        help='training-free: Res2CLIP* (no adapters) | finetune: Res2CLIP† (with adapters)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned checkpoint (.pth), required for --mode finetune')

    # Data
    parser.add_argument('--dataset',    type=str, default='mvtec', choices=['mvtec', 'visa', 'btad', ...])
    parser.add_argument('--data_path',  type=str, default='./data/MVTec')
    parser.add_argument('--save_path',  type=str, default='./results/')
    parser.add_argument('--image_size', type=int, default=518)

    # Feature layers
    parser.add_argument('--text_features_list',   type=int, nargs='+', default=[24])
    parser.add_argument('--visual_features_list', type=int, nargs='+', default=[6, 12, 18, 24])
    parser.add_argument('--res_features_list',    type=int, nargs='+', default=[24])

    # Few-shot memory
    parser.add_argument('--k_shot',  type=int, default=1)
    parser.add_argument('--seed',    type=int, default=111,
                        help='Seed for reference selection (ignored if --ref_json is set)')
    parser.add_argument('--shuffle', action='store_true', 
                        help='Shuffle training set before selecting references')
    parser.add_argument('--ref_json', type=str, default=None,
                        help='JSON {cls: [img_path, ...]} for exact reference reproducibility')

    # Matching
    parser.add_argument('--match_strategy',  type=str,  default='sparse_radial_knn')
    parser.add_argument('--spatial_penalty', type=float, default=0.01)
    parser.add_argument('--use_sparse',      action='store_true') # Redundant

    args = parser.parse_args()

    if args.mode == 'finetune' and args.checkpoint is None:
        parser.error('--checkpoint is required when --mode finetune')

    setup_seed(args.seed)
    test(args)
