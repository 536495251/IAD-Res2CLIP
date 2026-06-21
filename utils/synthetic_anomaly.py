"""
Synthetic anomaly generation for training on normal-only datasets.

Methods:
  - cut_paste:         Crop a patch from the image and paste elsewhere
  - cut_paste_jitter:  CutPaste + color jitter on pasted patch
  - perlin_noise:      Perlin noise mask with irregular defect shapes
  - local_blur:        Local Gaussian blur (simulates contamination/defocus)
  - scratch:           Thin line scratches
  - color_shift:       Local color shift (simulates oxidation/discoloration)
  - texture_replace:   Replace region with random noise texture
  - mixed:             Randomly select one of the above methods

All functions operate on normalized tensor images [C, H, W] or [B, C, H, W].
"""

import torch
import torch.nn.functional as F
import random
import math
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────

def _rand_patch(H, W, patch_ratio_range=(0.05, 0.15)):
    """Generate a random rectangular patch."""
    area = H * W
    target_area = random.uniform(*patch_ratio_range) * area
    aspect_ratio = random.uniform(0.5, 2.0)

    patch_h = int(round(math.sqrt(target_area * aspect_ratio)))
    patch_w = int(round(math.sqrt(target_area / aspect_ratio)))
    patch_h = min(patch_h, H - 2)
    patch_w = min(patch_w, W - 2)
    if patch_h < 4 or patch_w < 4:
        patch_h, patch_w = max(4, H // 8), max(4, W // 8)

    src_y = random.randint(0, H - patch_h - 1)
    src_x = random.randint(0, W - patch_w - 1)
    dst_y = random.randint(0, H - patch_h - 1)
    dst_x = random.randint(0, W - patch_w - 1)

    return (src_y, src_y + patch_h, src_x, src_x + patch_w,
            dst_y, dst_y + patch_h, dst_x, dst_x + patch_w)


def _perlin_noise_2d(H, W, scale=20, device='cpu'):
    """Generate 2D Perlin-like noise mask in [0, 1].

    Uses a simple approach: generate low-res noise, upsample with bicubic.
    """
    small_h = max(4, H // scale)
    small_w = max(4, W // scale)
    # Random noise at low resolution
    noise_low = torch.rand(1, 1, small_h, small_w, device=device)
    # Upsample to full resolution
    noise = F.interpolate(noise_low, size=(H, W), mode='bicubic', align_corners=False)
    noise = noise.squeeze(0).squeeze(0)
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


# ── Synthetic anomaly methods ────────────────────────────────────────

def cut_paste(image, patch_ratio_range=(0.05, 0.15)):
    """Copy a patch from source to destination location."""
    C, H, W = image.shape
    src_y1, src_y2, src_x1, src_x2, dst_y1, dst_y2, dst_x1, dst_x2 = _rand_patch(H, W, patch_ratio_range)

    augmented = image.clone()
    augmented[:, dst_y1:dst_y2, dst_x1:dst_x2] = image[:, src_y1:src_y2, src_x1:src_x2]

    mask = torch.zeros((1, H, W), device=image.device, dtype=image.dtype)
    mask[:, dst_y1:dst_y2, dst_x1:dst_x2] = 1.0
    return augmented, mask


def cut_paste_jitter(image, patch_ratio_range=(0.05, 0.15), brightness=0.3, contrast=0.3, saturation=0.3):
    """CutPaste with color jitter on the pasted patch."""
    C, H, W = image.shape
    src_y1, src_y2, src_x1, src_x2, dst_y1, dst_y2, dst_x1, dst_x2 = _rand_patch(H, W, patch_ratio_range)

    patch = image[:, src_y1:src_y2, src_x1:src_x2].clone()

    # Color jitter
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        patch = patch * factor
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = patch.mean(dim=(1, 2), keepdim=True)
        patch = (patch - mean) * factor + mean
    if saturation > 0:
        factor = 1.0 + random.uniform(-saturation, saturation)
        gray = patch.mean(dim=0, keepdim=True)
        patch = patch * factor + gray * (1 - factor)

    augmented = image.clone()
    augmented[:, dst_y1:dst_y2, dst_x1:dst_x2] = patch

    mask = torch.zeros((1, H, W), device=image.device, dtype=image.dtype)
    mask[:, dst_y1:dst_y2, dst_x1:dst_x2] = 1.0
    return augmented, mask


def perlin_noise(image, patch_ratio_range=(0.05, 0.2)):
    """Irregular-shaped anomaly using Perlin noise mask + color jitter.

    Simulates: dents, cracks, stains with natural-looking boundaries.
    """
    C, H, W = image.shape
    device = image.device

    # Perlin noise mask
    noise = _perlin_noise_2d(H, W, scale=random.randint(10, 40), device=device)

    # Threshold to get irregular region
    threshold = random.uniform(0.4, 0.7)
    binary_mask = (noise > threshold).float()

    # Optionally erode/dilate for smaller defects
    if random.random() < 0.3:
        kernel_size = random.randint(3, 7)
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size ** 2)
        binary_mask = binary_mask.unsqueeze(0).unsqueeze(0)
        binary_mask = F.conv2d(binary_mask, kernel, padding=kernel_size // 2).squeeze()
        binary_mask = (binary_mask > 0.5).float()

    # Ensure minimum defect size
    if binary_mask.sum() < H * W * 0.005:
        binary_mask[random.randint(0, H - 1), random.randint(0, W - 1)] = 1.0

    # Create anomaly: color jitter within the mask region
    augmented = image.clone()
    anomaly = image.clone()

    # Apply random color transformation to the anomaly region
    brightness_f = 1.0 + random.uniform(-0.4, 0.4)
    contrast_f = 1.0 + random.uniform(-0.4, 0.4)
    anomaly = anomaly * brightness_f
    mean = anomaly.mean(dim=(1, 2), keepdim=True)
    anomaly = (anomaly - mean) * contrast_f + mean
    anomaly = torch.clamp(anomaly, image.min().item(), image.max().item())

    # Ensure mask matches image size (safe guard for rounding errors)
    if binary_mask.shape[0] != H or binary_mask.shape[1] != W:
        binary_mask = F.interpolate(
            binary_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0).squeeze(0)

    # Blend anomaly into mask region
    mask_3ch = binary_mask.unsqueeze(0).expand(C, -1, -1)
    augmented = augmented * (1 - mask_3ch) + anomaly * mask_3ch

    mask = binary_mask.unsqueeze(0)
    return augmented, mask


def local_blur(image, patch_ratio_range=(0.05, 0.2)):
    """Local Gaussian blur — simulates surface contamination, defocus.

    Blurs a patch region using a large Gaussian kernel.
    """
    C, H, W = image.shape

    # Generate irregular mask via Perlin
    noise = _perlin_noise_2d(H, W, scale=random.randint(10, 30), device=image.device)
    threshold = random.uniform(0.5, 0.7)
    binary_mask = (noise > threshold).float()

    # Heavier blur
    k = random.choice([9, 11, 15, 21])
    sigma = random.uniform(2.0, 5.0)
    gaussian = _create_gaussian_kernel(k, sigma, C, device=image.device)

    # Apply blur to full image
    blurred = image.unsqueeze(0)
    blurred = gaussian(blurred).squeeze(0)

    # Blend only in mask region
    mask_3ch = binary_mask.unsqueeze(0).expand(C, -1, -1)
    augmented = image.clone()
    augmented = augmented * (1 - mask_3ch) + blurred * mask_3ch

    mask = binary_mask.unsqueeze(0)
    return augmented, mask


def scratch(image):
    """Thin line scratch — simulates scratches, cuts, surface damage."""
    C, H, W = image.shape
    device = image.device

    # Generate a line
    x1, y1 = random.randint(0, W - 1), random.randint(0, H - 1)
    length = random.randint(int(H * 0.1), int(H * 0.4))
    angle = random.uniform(0, 2 * math.pi)

    x2 = min(W - 1, max(0, int(x1 + length * math.cos(angle))))
    y2 = min(H - 1, max(0, int(y1 + length * math.sin(angle))))

    # Rasterize line
    n_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
    xs = torch.linspace(x1, x2, n_points, device=device).long()
    ys = torch.linspace(y1, y2, n_points, device=device).long()

    # Line width
    width = random.randint(1, 3)
    mask = torch.zeros((H, W), device=device)
    for dx in range(-width, width + 1):
        for dy in range(-width, width + 1):
            y_idx = (ys + dy).clamp(0, H - 1)
            x_idx = (xs + dx).clamp(0, W - 1)
            mask[y_idx, x_idx] = 1.0

    # Scratch appearance: bright or dark line
    augmented = image.clone()
    if random.random() < 0.5:
        scratch_val = image.max() + random.uniform(0.3, 0.8)  # bright scratch
    else:
        scratch_val = image.min() - random.uniform(0.3, 0.8)  # dark scratch
    scratch_val = torch.clamp(torch.tensor(scratch_val), -3.0, 3.0)

    mask_3ch = mask.unsqueeze(0).expand(C, -1, -1)
    augmented = augmented * (1 - mask_3ch) + scratch_val * mask_3ch

    mask = mask.unsqueeze(0)
    return augmented, mask


def color_shift(image, patch_ratio_range=(0.05, 0.15)):
    """Local color shift — simulates oxidation, discoloration, overheating.

    Applies a strong hue shift to a local region.
    """
    C, H, W = image.shape
    dst_y1, dst_y2, dst_x1, dst_x2, _, _, _, _ = _rand_patch(H, W, patch_ratio_range)

    mask = torch.zeros((1, H, W), device=image.device, dtype=image.dtype)
    mask[:, dst_y1:dst_y2, dst_x1:dst_x2] = 1.0

    augmented = image.clone()
    patch = augmented[:, dst_y1:dst_y2, dst_x1:dst_x2].clone()

    # Strong color shift: swap or scale channels
    if random.random() < 0.5:
        # Channel-wise scaling (different per channel)
        scales = torch.tensor([random.uniform(0.3, 1.7) for _ in range(3)],
                              device=image.device).view(3, 1, 1)
        patch = patch * scales
    else:
        # Channel shuffle (simulates severe chemical discoloration)
        perm = torch.randperm(3)
        patch = patch[perm]

    augmented[:, dst_y1:dst_y2, dst_x1:dst_x2] = patch
    return augmented, mask


def texture_replace(image, patch_ratio_range=(0.05, 0.2)):
    """Replace a region with random noise texture — simulates surface damage."""
    C, H, W = image.shape
    device = image.device

    # Generate irregular mask
    noise = _perlin_noise_2d(H, W, scale=random.randint(8, 25), device=device)
    threshold = random.uniform(0.4, 0.6)
    binary_mask = (noise > threshold).float()
    if binary_mask.sum() < H * W * 0.01:
        binary_mask = torch.ones((H, W), device=device)
        binary_mask[random.randint(0, H - 1), random.randint(0, W - 1)] = 0.0
        binary_mask = 1 - binary_mask

    # Generate random texture
    texture_type = random.choice(['uniform', 'gaussian', 'perlin'])
    if texture_type == 'uniform':
        texture = torch.rand(C, H, W, device=device) * 2 - 1
    elif texture_type == 'gaussian':
        texture = torch.randn(C, H, W, device=device) * 0.5
    else:  # perlin
        pn = _perlin_noise_2d(H, W, scale=random.randint(5, 15), device=device)
        texture = pn.unsqueeze(0).expand(C, -1, -1) * 0.5 + torch.randn(C, 1, 1, device=device) * 0.3

    # Blend with original statistics
    texture = texture - texture.mean()
    texture = texture / (texture.std() + 1e-8)
    texture = texture * image.std() * random.uniform(0.5, 1.5) + image.mean()

    mask_3ch = binary_mask.unsqueeze(0).expand(C, -1, -1)
    augmented = image.clone()
    augmented = augmented * (1 - mask_3ch) + texture * mask_3ch

    mask = binary_mask.unsqueeze(0)
    return augmented, mask


# ── Helpers ──────────────────────────────────────────────────────────

def _create_gaussian_kernel(kernel_size, sigma, channels, device='cpu'):
    """Create a Gaussian blur Conv2d layer."""
    x = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
    blur = torch.nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2,
                           groups=channels, bias=False, device=device)
    blur.weight.data = kernel_2d
    blur.weight.requires_grad = False
    return blur


# ── All methods registry ─────────────────────────────────────────────

ALL_METHODS = {
    'cut_paste': cut_paste,
    'cut_paste_jitter': cut_paste_jitter,
    'perlin_noise': perlin_noise,
    'local_blur': local_blur,
    'scratch': scratch,
    'color_shift': color_shift,
    'texture_replace': texture_replace,
}


# ── Batch augmentation ────────────────────────────────────────────────

def batch_augment(images, anomaly_ratio=0.5, method='mixed'):
    """Apply synthetic anomaly to a batch of images.

    Args:
        images:        [B, C, H, W] normalized tensor batch
        anomaly_ratio: fraction of batch to augment (e.g. 0.5 = half the batch)
        method:        'mixed' = random per-image, or specific method name

    Returns:
        aug_images:    [B, C, H, W]
        aug_masks:     [B, 1, H, W] binary masks (0 for unaugmented)
        anomaly_flags: [B] boolean
    """
    B, C, H, W = images.shape
    aug_images = images.clone()
    aug_masks = torch.zeros((B, 1, H, W), device=images.device, dtype=images.dtype)
    anomaly_flags = torch.zeros(B, dtype=torch.bool, device=images.device)

    n_aug = max(1, int(B * anomaly_ratio))
    aug_indices = random.sample(range(B), n_aug)

    for idx in aug_indices:
        if method == 'mixed':
            fn_name = random.choice(list(ALL_METHODS.keys()))
        else:
            fn_name = method
        fn = ALL_METHODS[fn_name]

        aug_img, mask = fn(images[idx])
        aug_images[idx] = aug_img
        aug_masks[idx] = mask
        anomaly_flags[idx] = True

    return aug_images, aug_masks, anomaly_flags
