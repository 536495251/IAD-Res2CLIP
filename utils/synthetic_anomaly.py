"""
Synthetic anomaly generation for training on normal-only datasets.

Methods:
  - cut_paste:    Crop a patch from the image and paste elsewhere (simple, fast)
  - cut_paste_jitter: CutPaste + color jitter on pasted patch (more varied)

All functions operate on normalized tensor images [C, H, W] or [B, C, H, W].
"""

import torch
import torch.nn.functional as F
import random
import math


def _rand_patch(H, W, patch_ratio_range=(0.05, 0.15)):
    """Generate a random patch (y1, y2, x1, x2) covering patch_ratio of the image."""
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


def cut_paste(image, patch_ratio_range=(0.05, 0.15)):
    """CutPaste: copy a patch from source to destination location.

    Args:
        image:       [C, H, W] normalized tensor
        patch_ratio_range: (min, max) fraction of image area for the patch

    Returns:
        augmented:   [C, H, W] image with synthetic anomaly
        mask:        [1, H, W] binary mask (1 = pasted region)
    """
    C, H, W = image.shape
    src_y1, src_y2, src_x1, src_x2, dst_y1, dst_y2, dst_x1, dst_x2 = _rand_patch(
        H, W, patch_ratio_range)

    augmented = image.clone()
    augmented[:, dst_y1:dst_y2, dst_x1:dst_x2] = image[:, src_y1:src_y2, src_x1:src_x2]

    mask = torch.zeros((1, H, W), device=image.device, dtype=image.dtype)
    mask[:, dst_y1:dst_y2, dst_x1:dst_x2] = 1.0

    return augmented, mask


def cut_paste_jitter(image, patch_ratio_range=(0.05, 0.15), brightness=0.3, contrast=0.3, saturation=0.3):
    """CutPaste with color jitter on the pasted patch.

    The jitter simulates appearance changes of real defects (discoloration, etc.).

    Args:
        image:       [C, H, W] normalized tensor
        patch_ratio_range: (min, max) fraction of image area
        brightness/contrast/saturation: jitter magnitude

    Returns:
        augmented:   [C, H, W]
        mask:        [1, H, W]
    """
    C, H, W = image.shape
    src_y1, src_y2, src_x1, src_x2, dst_y1, dst_y2, dst_x1, dst_x2 = _rand_patch(
        H, W, patch_ratio_range)

    patch = image[:, src_y1:src_y2, src_x1:src_x2].clone()

    # Color jitter on the patch
    # Brightness
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        patch = patch * factor
    # Contrast
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = patch.mean(dim=(1, 2), keepdim=True)
        patch = (patch - mean) * factor + mean
    # Saturation (approximate via channel-wise scaling)
    if saturation > 0:
        factor = 1.0 + random.uniform(-saturation, saturation)
        gray = patch.mean(dim=0, keepdim=True)
        patch = patch * factor + gray * (1 - factor)

    augmented = image.clone()
    augmented[:, dst_y1:dst_y2, dst_x1:dst_x2] = patch

    mask = torch.zeros((1, H, W), device=image.device, dtype=image.dtype)
    mask[:, dst_y1:dst_y2, dst_x1:dst_x2] = 1.0

    return augmented, mask


def batch_augment(images, anomaly_ratio=0.5, method='cut_paste_jitter'):
    """Apply synthetic anomaly to a batch of images.

    Args:
        images:        [B, C, H, W] normalized tensor batch
        anomaly_ratio: fraction of batch to augment (e.g. 0.5 = half the batch)
        method:        'cut_paste' or 'cut_paste_jitter'

    Returns:
        aug_images:    [B, C, H, W] (some augmented, some original)
        aug_masks:     [B, 1, H, W] binary masks (0 for unaugmented)
        anomaly_flags: [B] boolean, True where augmented
    """
    B, C, H, W = images.shape
    aug_images = images.clone()
    aug_masks = torch.zeros((B, 1, H, W), device=images.device, dtype=images.dtype)
    anomaly_flags = torch.zeros(B, dtype=torch.bool, device=images.device)

    n_aug = max(1, int(B * anomaly_ratio))
    aug_indices = random.sample(range(B), n_aug)

    fn = cut_paste_jitter if method == 'cut_paste_jitter' else cut_paste

    for idx in aug_indices:
        aug_img, mask = fn(images[idx])
        aug_images[idx] = aug_img
        aug_masks[idx] = mask
        anomaly_flags[idx] = True

    return aug_images, aug_masks, anomaly_flags
