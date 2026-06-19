import torch
import torch.nn.functional as F
import math
import re


def sparsemax_lookup(sim_matrix):
    '''Sparsemax normalization for adaptive sparse matching (Martins & Astudillo, 2016).

    Args:
        sim_matrix: [B, L, N] — similarity matrix
    Returns:
        sparse_weights: [B, L, N] — truncated non-negative weights summing to 1,
                        many entries are exactly zero
    '''
    dim = -1
    number_of_logits = sim_matrix.size(dim)

    # Subtract max for numerical stability
    sim_max, _ = torch.max(sim_matrix, dim=dim, keepdim=True)
    z = sim_matrix - sim_max

    # Sort descending to find threshold
    zs, _ = torch.sort(z, dim=dim, descending=True)

    # Build index tensor [1, 2, ..., N]
    arange = torch.arange(1, number_of_logits + 1, device=sim_matrix.device, dtype=sim_matrix.dtype)
    arange = arange.view(1, 1, -1)

    # Find the largest k satisfying the sparsemax boundary condition
    bound = 1.0 + arange * zs
    cumulative_sum_zs = torch.cumsum(zs, dim=dim)
    is_gt = torch.gt(bound, cumulative_sum_zs).type(sim_matrix.type())
    k = torch.max(is_gt * arange, dim=dim, keepdim=True)[0]

    # Compute threshold tau and truncate
    taus = (torch.sum(is_gt * zs, dim=dim, keepdim=True) - 1.0) / k
    return torch.max(torch.zeros_like(z), z - taus)


def _compute_penalty(base_strategy, H_feat, k_shot, spatial_penalty, device):
    '''Compute the [L, k_shot*L] spatial penalty matrix for a given strategy.

    Supported base strategies (the part of match_strategy before the "_Xnn" suffix):

        ""               — no penalty
        "local"          — L2 Euclidean distance between patch positions
        "radial"         — |r_q − r_ref|, radial ring distance from image centre (paper Eq. 8)
        "radial_new"     — radial distance normalized by the max possible radial distance
        "radial_margin"  — radial with a 2-patch dead-zone before penalising
        "radial_laplace" — exponential radial
        "polar_linear"   — radial + angular distance
        "polar"          — radial (Gaussian) + angular (Gaussian)
        "radial_euc"     — normalized radial + normalized Euclidean
        "euc_new"        — Euclidean with 2-patch dead-zone + exponential saturation

    All strategies scale proportionally with spatial_penalty (γ).
    To add a new strategy, add an elif branch here — no other changes needed.
    '''
    if not base_strategy:
        return 0.0

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H_feat), torch.arange(H_feat), indexing='ij')
    cy, cx = (H_feat - 1) / 2.0, (H_feat - 1) / 2.0
    y = grid_y.flatten().float().to(device) - cy
    x = grid_x.flatten().float().to(device) - cx
    r = torch.sqrt(x ** 2 + y ** 2)
    rad_dist = torch.abs(r.unsqueeze(1) - r.unsqueeze(0)).repeat(1, k_shot)  # [L, k_shot*L]

    if base_strategy == "local":
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float().to(device)
        euc_dist = torch.cdist(coords, coords, p=2).repeat(1, k_shot)
        return spatial_penalty * euc_dist

    elif base_strategy == "radial":
        return spatial_penalty * rad_dist

    elif base_strategy == "radial_new":
        max_rad = math.sqrt(2.0) * ((H_feat - 1) / 2.0)
        return spatial_penalty * rad_dist / (max_rad + 1e-6)

    elif base_strategy == "radial_margin":
        return spatial_penalty * torch.clamp(rad_dist - 2.0, min=0.0)

    elif base_strategy == "radial_laplace":
        sigma_r = 1.0
        return spatial_penalty * (1.0 - torch.exp(-rad_dist / sigma_r))

    elif base_strategy == "polar_linear":
        theta = torch.atan2(y, x)
        theta_dist = torch.abs(theta.unsqueeze(1) - theta.unsqueeze(0))
        theta_dist = torch.min(theta_dist, 2 * math.pi - theta_dist).repeat(1, k_shot)
        return spatial_penalty * (rad_dist + theta_dist)

    elif base_strategy == "polar":
        theta = torch.atan2(y, x)
        theta_dist = torch.abs(theta.unsqueeze(1) - theta.unsqueeze(0))
        theta_dist = torch.min(theta_dist, 2 * math.pi - theta_dist).repeat(1, k_shot)
        sigma_r, sigma_theta = 1.0, math.pi / 4.0
        p_r = spatial_penalty * (1.0 - torch.exp(-(rad_dist ** 2) / (2 * sigma_r ** 2)))
        p_t = spatial_penalty * 0.1 * (1.0 - torch.exp(-(theta_dist ** 2) / (2 * sigma_theta ** 2)))
        return p_r + p_t

    elif base_strategy == "radial_euc":
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float().to(device)
        euc_dist = torch.cdist(coords, coords, p=2).repeat(1, k_shot)
        max_euc = math.sqrt(2.0) * (H_feat - 1)
        max_rad = math.sqrt(2.0) * ((H_feat - 1) / 2.0)
        return spatial_penalty * (rad_dist / (max_rad + 1e-6) + euc_dist / (max_euc + 1e-6))

    elif base_strategy == "euc_new":
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float().to(device)
        euc_dist = torch.cdist(coords, coords, p=2).repeat(1, k_shot)
        effective_euc = torch.clamp(euc_dist - 2.0, min=0.0)
        return spatial_penalty * (1.0 - torch.exp(-effective_euc))

    else:
        valid = ("local, radial, radial_new, radial_margin, radial_laplace, "
                 "polar_linear, polar, radial_euc, euc_new")
        raise ValueError(f"Unknown spatial penalty strategy '{base_strategy}'. Valid: {valid}")


def get_visual_match(feat, ref_feat, strategy, spatial_penalty, k_shot, device,
                     pooling=False, k_neighbors=5, base_temp=0.05, use_sparse=False):
    '''Retrieve spatially-aligned reference features for each query patch via KNN matching.

    Strategy string format: ["sparse_"] <penalty> "_" <K> "nn"
        - prefix "sparse_"  enables sparsemax aggregation instead of softmax
        - <penalty>         spatial penalty type (see _compute_penalty for all options)
        - <K>               number of neighbors; "k" uses the k_neighbors argument
        Examples: "sparse_radial_knn", "2nn", "radial_3nn", "polar_linear_2nn"

    Args:
        feat:            query patch features [B, L, D], L2-normalized
        ref_feat:        reference patch features [1, k_shot*L, D], L2-normalized
        strategy:        matching strategy string
        spatial_penalty: weight γ for the spatial penalty
        k_shot:          number of reference images
        device:          torch device
        pooling:         if True, smooth features with 3×3 avg pool before matching
        k_neighbors:     K for KNN when strategy uses "knn" (i.e., K not specified in string)
        base_temp:       softmax temperature for KNN aggregation
        use_sparse:      if True, force sparsemax regardless of strategy prefix
    Returns:
        map_vis:     pixel-level visual anomaly map [B, 1, H, H]
        matched_ref: spatially-aligned reference feature [B, L, D], L2-normalized
    '''
    B, L, D = feat.shape
    H_feat = int(L ** 0.5)

    if pooling:
        # Smooth query and reference features with 3×3 avg pool for more robust addressing
        feat_2d = feat.permute(0, 2, 1).view(B, D, H_feat, H_feat)
        feat_2d = F.pad(feat_2d, (1, 1, 1, 1), mode='replicate')
        feat_search = F.avg_pool2d(feat_2d, kernel_size=3, stride=1, padding=0)
        feat_search = feat_search.view(B, D, L).permute(0, 2, 1)
        feat_search = feat_search / feat_search.norm(dim=-1, keepdim=True)

        ref_2d = ref_feat.view(k_shot, L, D).permute(0, 2, 1).view(k_shot, D, H_feat, H_feat)
        ref_2d = F.pad(ref_2d, (1, 1, 1, 1), mode='replicate')
        ref_search = F.avg_pool2d(ref_2d, kernel_size=3, stride=1, padding=0)
        ref_search = ref_search.view(k_shot, D, L).permute(0, 2, 1).reshape(1, k_shot * L, D)
        ref_search = ref_search / ref_search.norm(dim=-1, keepdim=True)

        sim_v = feat @ ref_feat.permute(0, 2, 1)
        sim_search = (feat_search @ ref_search.permute(0, 2, 1) + sim_v) / 2
    else:
        sim_v = feat @ ref_feat.permute(0, 2, 1)   # [B, L, k_shot*L]
        sim_search = sim_v

    # Parse strategy string: optional "sparse_" prefix + penalty type + K suffix
    if strategy.startswith("sparse_"):
        use_sparse = True
        strategy = strategy[7:]

    match = re.search(r'_?([kK]|\d+)nn$', strategy)
    if not match:
        raise ValueError(f"Invalid match_strategy '{strategy}'. Must end with 'Xnn' or 'knn'.")

    k_str = match.group(1)
    if k_str.lower() != 'k':
        k_neighbors = int(k_str)

    if k_neighbors < 1:
        raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}.")

    is_knn = k_neighbors > 1
    base_strategy = strategy[:match.start()]

    penalty = _compute_penalty(base_strategy, H_feat, k_shot, spatial_penalty, device)

    if isinstance(penalty, torch.Tensor):
        sim_v_penalized = sim_search - penalty.unsqueeze(0)
    else:
        sim_v_penalized = sim_search

    # Aggregate matched reference features
    if use_sparse:
        # Sparsemax: adaptive sparse weighting, entries below threshold become exactly zero
        attn_weights = sparsemax_lookup(sim_v_penalized)       # [B, L, k_shot*L]
        matched_ref = torch.bmm(attn_weights, ref_feat.expand(B, -1, -1))
        max_sim = (sim_v * attn_weights).sum(dim=-1)

    elif is_knn:
        # KNN: softmax-weighted average of the K nearest neighbours
        topk_sim_penalized, topk_idx = sim_v_penalized.topk(k_neighbors, dim=-1)
        topk_sim_true = torch.gather(sim_v, dim=-1, index=topk_idx)
        attn_weights = F.softmax(topk_sim_true / base_temp, dim=-1)

        ref_feat_b = ref_feat.expand(B, -1, -1)
        flat_idx = topk_idx.view(B, L * k_neighbors).unsqueeze(-1).expand(-1, -1, D)
        matched_ref_k = torch.gather(ref_feat_b, dim=1, index=flat_idx).view(B, L, k_neighbors, D)
        matched_ref = (matched_ref_k * attn_weights.unsqueeze(-1)).sum(dim=2)
        max_sim = (topk_sim_true * attn_weights).sum(dim=-1)

    else:
        # 1-NN: single nearest neighbour
        max_idx = sim_v_penalized.argmax(dim=-1)
        max_sim = torch.gather(sim_v, dim=-1, index=max_idx.unsqueeze(-1)).squeeze(-1)
        matched_ref = torch.gather(ref_feat.expand(B, -1, -1), dim=1,
                                   index=max_idx.unsqueeze(-1).expand(-1, -1, D))

    matched_ref = matched_ref / matched_ref.norm(dim=-1, keepdim=True)
    max_sim = (feat * matched_ref).sum(dim=-1)
    map_vis = (1.0 - max_sim).view(B, 1, H_feat, H_feat)

    return map_vis, matched_ref
