import torch
import torch.nn.functional as F
import math


def loss_dynamic_margin_ranking(score, R_base, gt_mask, tau=0.3):
    B = score.shape[0]
    H_feat = int(score.shape[1] ** 0.5)
    score_map = score.view(B, 1, H_feat, H_feat)
    score_map = F.interpolate(score_map, size=gt_mask.shape[-2:], mode='bilinear')
    mag = torch.norm(R_base.detach(), p=2, dim=-1)
    margin_map = mag.view(B, 1, H_feat, H_feat)
    margin_map = F.interpolate(margin_map, size=gt_mask.shape[-2:], mode='bilinear') * tau
    num_n = torch.sum(1 - gt_mask).clamp(min=1e-6)
    num_a = torch.sum(gt_mask).clamp(min=1e-6)
    loss_n = torch.sum(F.relu(score_map) * (1 - gt_mask)) / num_n
    loss_a = torch.sum(F.relu(margin_map - score_map) * gt_mask) / num_a
    return loss_n + loss_a


def loss_dynamic_margin_ranking_global(score_global, R_base_global, gt_label, tau=0.3):
    num_n = torch.sum(1 - gt_label).clamp(min=1e-6)
    num_a = torch.sum(gt_label).clamp(min=1e-6)
    margin = tau * torch.norm(R_base_global.detach(), p=2, dim=-1)
    loss_n = torch.sum(F.relu(score_global) * (1 - gt_label)) / num_n
    loss_a = torch.sum(F.relu(margin - score_global) * gt_label) / num_a
    return loss_n + loss_a


def loss_magnitude_control(R_adapt, R_ori, gt_mask):
    B, L, D = R_adapt.shape
    H_feat = int(math.sqrt(L))
    num_n = torch.sum(1 - gt_mask).clamp(min=1e-6)
    num_a = torch.sum(gt_mask).clamp(min=1e-6)
    mag_adapt = torch.norm(R_adapt, p=2, dim=-1).view(B, 1, H_feat, H_feat)
    mag_ori   = torch.norm(R_ori.detach(), p=2, dim=-1).view(B, 1, H_feat, H_feat)
    mag_adapt = F.interpolate(mag_adapt, size=gt_mask.shape[-2:], mode='bilinear')
    mag_ori   = F.interpolate(mag_ori,   size=gt_mask.shape[-2:], mode='bilinear')
    loss_n = torch.sum(mag_adapt * (1 - gt_mask)) / num_n
    loss_a = torch.sum(torch.abs(mag_adapt - mag_ori) * gt_mask) / num_a
    return loss_n, loss_a


def loss_fixed_margin_ranking(score, gt_mask, margin=0.3):
    B = score.shape[0]
    H_feat = int(score.shape[1] ** 0.5)
    score_map = score.view(B, 1, H_feat, H_feat)
    score_map = F.interpolate(score_map, size=gt_mask.shape[-2:], mode='bilinear')
    num_n = torch.sum(1 - gt_mask).clamp(min=1e-6)
    num_a = torch.sum(gt_mask).clamp(min=1e-6)
    loss_n = torch.sum(F.relu(score_map) * (1 - gt_mask)) / num_n
    loss_a = torch.sum(F.relu(margin - score_map) * gt_mask) / num_a
    return loss_n + loss_a


def loss_fixed_margin_ranking_global(score_global, gt_label, margin=0.3):
    num_n = torch.sum(1 - gt_label).clamp(min=1e-6)
    num_a = torch.sum(gt_label).clamp(min=1e-6)
    loss_n = torch.sum(F.relu(score_global) * (1 - gt_label)) / num_n
    loss_a = torch.sum(F.relu(margin - score_global) * gt_label) / num_a
    return loss_n + loss_a


