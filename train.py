import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import math
import random
import itertools
import numpy as np
from tqdm import tqdm
import warnings

from models.clip import load_clean_clip, TextFeatureBank
from models.adapter import ResidualAdapter, MultiScaleConvAdapter
from data.dataset import RefDataset
from utils.utils import get_transform, get_cosine_schedule_with_warmup
from utils.loss import (loss_dynamic_margin_ranking, loss_dynamic_margin_ranking_global,
                        loss_magnitude_control, loss_fixed_margin_ranking,
                        loss_fixed_margin_ranking_global)
from models.knn import get_visual_match

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train(args):
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_path, exist_ok=True)

    model, _ = load_clean_clip('ViT-L/14@336px', device=device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    preprocess, target_transform = get_transform(args)
    train_data = RefDataset(
        root=args.data_path, transform=preprocess, target_transform=target_transform,
        dataset_name=args.dataset, mode=args.data_mode,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
    )

    text_bank = TextFeatureBank(model, device)

    TEXT_LAYERS = args.text_features_list
    VIS_LAYERS  = args.visual_features_list
    RES_LAYERS  = args.res_features_list
    ALL_LAYERS  = sorted(set(TEXT_LAYERS + VIS_LAYERS + RES_LAYERS))

    # Visual branch: {A_{v,l}} — one adapter per visual layer
    vis_adapters = nn.ModuleDict({str(l): MultiScaleConvAdapter() for l in VIS_LAYERS}).to(device)
    # Residual branch: {A^res_{v,l}} — one adapter per residual layer
    res_vis_local_adapters = nn.ModuleDict({str(l): MultiScaleConvAdapter() for l in RES_LAYERS}).to(device)
    # Residual branch: A^res_{v,cls} — global visual adapter
    res_vis_global_adapter = ResidualAdapter().to(device)
    # Residual branch: A^res_t — text residual adapter
    res_text_adapter = ResidualAdapter().to(device)
    # Text branch: A_t — text residual adapter
    text_adapter = ResidualAdapter().to(device)

    opt_vis      = torch.optim.Adam(vis_adapters.parameters(), lr=args.lr_v, betas=(0.5, 0.999))
    opt_res_vis  = torch.optim.Adam(
        list(res_vis_local_adapters.parameters()) + list(res_vis_global_adapter.parameters()),
        lr=args.lr_v, betas=(0.5, 0.999),
    )
    opt_res_text = torch.optim.Adam(res_text_adapter.parameters(), lr=args.lr_t, betas=(0.5, 0.999))
    opt_text     = torch.optim.Adam(text_adapter.parameters(),     lr=args.lr_t, betas=(0.5, 0.999))

    steps    = len(train_loader)
    active_v = math.ceil(args.epochs / 2.0)
    active_t = args.epochs // 2

    sched_vis      = get_cosine_schedule_with_warmup(opt_vis,      steps, steps * args.epochs)
    sched_res_vis  = get_cosine_schedule_with_warmup(opt_res_vis,  steps, steps * active_v)
    sched_res_text = get_cosine_schedule_with_warmup(opt_res_text, steps, steps * max(1, active_t))
    sched_text     = get_cosine_schedule_with_warmup(opt_text,     steps, steps * args.epochs)

    for epoch in range(args.epochs):
        train_res_vis = (epoch % 2 == 0)

        vis_adapters.train()
        for p in vis_adapters.parameters():
            p.requires_grad = True
        text_adapter.train()
        for p in text_adapter.parameters():
            p.requires_grad = True

        if train_res_vis:
            print(f"\n[Epoch {epoch+1}/{args.epochs}] Training: residual branch (visual adapters)")
            res_vis_local_adapters.train()
            res_vis_global_adapter.train()
            res_text_adapter.eval()
            for p in itertools.chain(res_vis_local_adapters.parameters(), res_vis_global_adapter.parameters()):
                p.requires_grad = True
            for p in res_text_adapter.parameters():
                p.requires_grad = False
        else:
            print(f"\n[Epoch {epoch+1}/{args.epochs}] Training: residual branch (text adapter)")
            res_vis_local_adapters.eval()
            res_vis_global_adapter.eval()
            res_text_adapter.train()
            for p in itertools.chain(res_vis_local_adapters.parameters(), res_vis_global_adapter.parameters()):
                p.requires_grad = False
            for p in res_text_adapter.parameters():
                p.requires_grad = True

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for items in pbar:
            image     = items['img'].to(device)
            ref_image = items['ref_img'].to(device)
            gt_mask   = items['img_mask'].to(device)
            gt_mask   = torch.where(gt_mask > 0.5, 1.0, 0.0)
            cls_names = items['cls_name']
            B = image.shape[0]

            opt_vis.zero_grad()
            opt_res_vis.zero_grad()
            opt_res_text.zero_grad()
            opt_text.zero_grad()

            loss_vis      = torch.tensor(0.0, device=device)
            loss_res_vis  = torch.tensor(0.0, device=device)
            loss_res_text = torch.tensor(0.0, device=device)
            loss_text     = torch.tensor(0.0, device=device)

            combined = torch.cat([image, ref_image], dim=0)
            with torch.no_grad():
                comb_feats_g, comb_feats_p = model.encode_image(combined, ALL_LAYERS, DPAM_layer=20)
                feat_g_q   = comb_feats_g[:B] / comb_feats_g[:B].norm(dim=-1, keepdim=True)
                feat_g_ref = comb_feats_g[B:] / comb_feats_g[B:].norm(dim=-1, keepdim=True)

                # R_t: raw text residual (frozen, class-agnostic)
                R_t = text_bank.get_features()[-1].unsqueeze(0).expand(B, -1)

            # R̃_t = A_t(R_t): adapted text residual for text branch
            R_t_text = text_adapter(R_t)
            R_t_text = R_t_text / R_t_text.norm(dim=-1, keepdim=True)

            # R̃^res_t = A^res_t(R_t): adapted text residual for residual branch
            if train_res_vis:
                with torch.no_grad():
                    R_t_res = res_text_adapter(R_t)
            else:
                R_t_res = res_text_adapter(R_t)
            R_t_res = R_t_res / R_t_res.norm(dim=-1, keepdim=True)

            gt_label_global = gt_mask.view(B, -1).max(dim=-1)[0]

            # ── Text branch optimization ─────────────────────────────────
            # Push ⟨f^q_v, R̃_t⟩ above margin for anomalies, below 0 for normals
            s_text_cls = (feat_g_q.detach() * R_t_text).sum(dim=-1)
            loss_text += loss_fixed_margin_ranking_global(s_text_cls, gt_label_global, margin=args.tau)
            loss_text += (1.0 - F.cosine_similarity(R_t_text, R_t.detach(), dim=-1)).mean() # Regularization

            # ── Global visual residual R_{v,cls} = f^q_v − f_ref ──────────────────
            R_v_cls = feat_g_q - feat_g_ref
            if train_res_vis:
                # R̃^res_{v,cls} = A^res_{v,cls}(R_{v,cls})
                R_v_cls_res = res_vis_global_adapter(R_v_cls)
                s_res_cls = (R_v_cls_res * R_t_res.detach()).sum(dim=-1)
                loss_res_vis += loss_dynamic_margin_ranking_global(
                    s_res_cls, R_v_cls.detach(), gt_label_global, tau=args.tau)
                loss_res_vis += F.l1_loss(R_v_cls_res.norm(dim=-1), R_v_cls.norm(dim=-1).detach()) # Regularization
            else:
                with torch.no_grad():
                    R_v_cls_res = res_vis_global_adapter(R_v_cls)
                s_res_cls = (R_v_cls_res.detach() * R_t_res).sum(dim=-1)
                loss_res_text += loss_dynamic_margin_ranking_global(
                    s_res_cls, R_v_cls.detach(), gt_label_global, tau=args.tau)
                loss_res_text += (1.0 - F.cosine_similarity(R_t_res, R_t.detach(), dim=-1)).mean() # Regularization

            # ── Patch-level branches ───────────────────────────────────────────────
            for i, layer_num in enumerate(ALL_LAYERS):
                l_key = str(layer_num)
                f_q   = comb_feats_p[i][:B][:, 1:, :]
                f_ref = comb_feats_p[i][B:][:, 1:, :]
                f_q   = f_q   / f_q.norm(dim=-1, keepdim=True)
                f_ref = f_ref / f_ref.norm(dim=-1, keepdim=True)

                matched_ref_list = []
                with torch.no_grad():
                    for b in range(B):
                        _, matched = get_visual_match(
                            feat=f_q[b:b+1], ref_feat=f_ref[b:b+1],
                            strategy=args.match_strategy, spatial_penalty=args.spatial_penalty,
                            k_shot=1, device=device,
                            use_sparse=args.use_sparse,
                        )
                        matched_ref_list.append(matched.squeeze(0))
                matched_ref = torch.stack(matched_ref_list, dim=0)

                # R_{v,l} = F̂^q_{v,l} − F̂_ref,l  (raw local visual residual)
                R_v = f_q - matched_ref

                # Visual branch: magnitude constraint on R̃_{v,l} = A_{v,l}(R_{v,l})
                if layer_num in VIS_LAYERS:
                    R_v_vis = vis_adapters[l_key](R_v)
                    l_n, l_a = loss_magnitude_control(R_v_vis, R_v.detach(), gt_mask.detach())
                    loss_vis += l_n + l_a

                # Text branch: patch-level cosine score
                if layer_num in TEXT_LAYERS:
                    s_text_patch = (f_q.detach() @ R_t_text.unsqueeze(-1)).squeeze(-1)
                    loss_text += loss_fixed_margin_ranking(s_text_patch, gt_mask.detach(), margin=args.tau)

                # Residual branch: R̃^res_{v,l} · R̃^res_t
                if train_res_vis and layer_num in RES_LAYERS:
                    # R̃^res_{v,l} = A^res_{v,l}(R_{v,l})
                    R_v_res = res_vis_local_adapters[l_key](R_v)
                    s_res_patch = (R_v_res @ R_t_res.detach().unsqueeze(-1)).squeeze(-1)
                    loss_res_vis += loss_dynamic_margin_ranking(
                        s_res_patch, R_v.detach(), gt_mask.detach(), tau=args.tau)
                    loss_res_vis += F.l1_loss(R_v_res.norm(dim=-1), R_v.norm(dim=-1).detach()) # Regularization
                elif not train_res_vis and layer_num in RES_LAYERS:
                    with torch.no_grad():
                        R_v_res = res_vis_local_adapters[l_key](R_v)
                    s_res_patch = (R_v_res.detach() @ R_t_res.unsqueeze(-1)).squeeze(-1)
                    loss_res_text += loss_dynamic_margin_ranking(
                        s_res_patch, R_v.detach(), gt_mask.detach(), tau=args.tau)

            loss_vis.backward()
            torch.nn.utils.clip_grad_norm_(vis_adapters.parameters(), 1.0)
            opt_vis.step()
            sched_vis.step()

            loss_text.backward()
            torch.nn.utils.clip_grad_norm_(text_adapter.parameters(), 1.0)
            opt_text.step()
            sched_text.step()

            if train_res_vis:
                loss_res_vis.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(res_vis_local_adapters.parameters()) + list(res_vis_global_adapter.parameters()), 1.0)
                opt_res_vis.step()
                sched_res_vis.step()
                pbar.set_postfix({'L_vis': f'{loss_vis.item():.3f}',
                                  'L_res_vis': f'{loss_res_vis.item():.3f}',
                                  'L_text': f'{loss_text.item():.3f}'})
            else:
                loss_res_text.backward()
                torch.nn.utils.clip_grad_norm_(res_text_adapter.parameters(), 1.0)
                opt_res_text.step()
                sched_res_text.step()
                pbar.set_postfix({'L_vis': f'{loss_vis.item():.3f}',
                                  'L_res_text': f'{loss_res_text.item():.3f}',
                                  'L_text': f'{loss_text.item():.3f}'})

            del combined, comb_feats_g, comb_feats_p
            torch.cuda.empty_cache()

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            torch.save({
                'vis_adapters': vis_adapters.state_dict(),
                'res_vis_local_adapters': res_vis_local_adapters.state_dict(),
                'res_vis_global_adapter': res_vis_global_adapter.state_dict(),
                'res_text_adapter': res_text_adapter.state_dict(),
                'text_adapter': text_adapter.state_dict(),
            }, os.path.join(args.save_path, f"checkpoint_ep{epoch+1}.pth"))
            print(f" > Checkpoint saved: epoch {epoch+1}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Res2CLIP Fine-tuning")
    parser.add_argument('--dataset',   type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--data_path', type=str, default='./data/MVTec')
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    parser.add_argument('--data_mode', type=str, default='test')
    parser.add_argument('--match_strategy',  type=str, default='sparse_radial_knn')
    parser.add_argument('--spatial_penalty', type=float, default=0.01)
    parser.add_argument('--use_sparse', action='store_true') # Redundant
    parser.add_argument('--text_features_list',   type=int, nargs='+', default=[24])
    parser.add_argument('--visual_features_list', type=int, nargs='+', default=[6, 12, 18, 24])
    parser.add_argument('--res_features_list',    type=int, nargs='+', default=[6, 12, 18, 24])
    parser.add_argument('--tau',        type=float, default=1.0)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--lr_v',       type=float, default=5e-4)
    parser.add_argument('--lr_t',       type=float, default=1e-4)
    parser.add_argument('--image_size', type=int,   default=518)
    parser.add_argument('--save_freq',  type=int,   default=10)
    parser.add_argument('--seed',       type=int,   default=111)
    args = parser.parse_args()
    print(args)
    train(args)
