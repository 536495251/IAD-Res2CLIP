from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np
import torch
import math

from adeval import EvalAccumulatorCuda


def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp,
                   use_metrics=['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)

    score_min = min(pr_sp)
    score_max = max(pr_sp)
    anomap_min = pr_px.min()
    anomap_max = pr_px.max()

    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=200)
    accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                           torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))

    metrics = accum.summary()
    metric_results = {}
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            metric_results[metric] = roc_auc_score(gt_sp, pr_sp)
        elif metric.startswith('I-AP'):
            metric_results[metric] = average_precision_score(gt_sp, pr_sp)
        elif metric.startswith('I-F1_max'):
            metric_results[metric] = f1_score_max(gt_sp, pr_sp)
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = metrics['p_auroc']
        elif metric.startswith('P-AP'):
            metric_results[metric] = metrics['p_aupr']
        elif metric.startswith('P-F1_max'):
            metric_results[metric] = f1_score_max(gt_px.ravel(), pr_px.ravel())
        elif metric.startswith('AUPRO'):
            metric_results[metric] = metrics['p_aupro']
    return list(metric_results.values())


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
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

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels, bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter
