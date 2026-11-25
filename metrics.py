"""Dataset-level classification, overlap, and boundary metrics."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch
from torch import Tensor

from model import Prediction


def boundary(mask: np.ndarray) -> np.ndarray:
    """Extract the inner boundary using four-connected erosion."""
    return mask & ~binary_erosion(mask, border_value=0)


def boundary_counts(pred: np.ndarray, target: np.ndarray, tolerance: float) -> tuple[int, int, int, int]:
    """Count matched and total boundary pixels in both matching directions."""
    p, t = boundary(pred), boundary(target)
    npred, ntarget = int(p.sum()), int(t.sum())
    if not npred or not ntarget:
        return 0, npred, 0, ntarget
    matched_p = int((p & (distance_transform_edt(~t) <= tolerance)).sum())
    matched_t = int((t & (distance_transform_edt(~p) <= tolerance)).sum())
    return matched_p, npred, matched_t, ntarget


class DiseaseMetrics:
    """Accumulate test-set counts before computing the four reported metrics."""

    def __init__(self, num_classes: int, threshold: float = 0.5, tolerance: int = 2) -> None:
        if num_classes < 2 or not 0 <= threshold <= 1 or tolerance < 0:
            raise ValueError("Invalid metric configuration.")
        self.num_classes = num_classes
        self.threshold = threshold
        self.tolerance = tolerance
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        self.pixel_confusion = torch.zeros(2, 2, dtype=torch.int64)
        self.boundary_totals = np.zeros(4, dtype=np.int64)

    @torch.no_grad()
    def update(self, output: Prediction, masks: Tensor, labels: Tensor) -> None:
        if (masks.ndim != 4 or masks.shape[1] != 1 or
                masks.shape != output.mask_logits.shape):
            raise ValueError("Mask and logit shape must match [B, 1, H, W].")
        if labels.shape != (masks.shape[0],) or output.class_logits.shape != (masks.shape[0], self.num_classes):
            raise ValueError("Class label and logit shape does not match the batch.")
        if labels.is_floating_point() or not ((labels >= 0) & (labels < self.num_classes)).all():
            raise ValueError("Class labels must be valid integer indices.")
        if not ((masks == 0) | (masks == 1)).all():
            raise ValueError("Masks must contain binary labels.")
        if not torch.isfinite(output.mask_logits).all() or not torch.isfinite(output.class_logits).all():
            raise FloatingPointError("Nonfinite evaluation logits.")
        pred_labels = output.class_logits.argmax(1).cpu()
        labels = labels.long().cpu()
        self.confusion += torch.bincount(labels * self.num_classes + pred_labels,
                                         minlength=self.num_classes ** 2).reshape(self.confusion.shape)
        pred = (output.mask_logits.float().sigmoid() >= self.threshold).cpu()
        target = masks.cpu().bool()
        self.pixel_confusion += torch.bincount((target.long() * 2 + pred.long()).flatten(),
                                               minlength=4).reshape(2, 2)
        for p, t in zip(pred[:, 0].numpy(), target[:, 0].numpy()):
            self.boundary_totals += boundary_counts(p, t, self.tolerance)

    def compute(self) -> dict[str, float]:
        if self.confusion.sum() == 0:
            raise ValueError("No evaluation samples have been accumulated.")
        cm = self.confusion.double()
        tp = cm.diag()
        denom = cm.sum(0) + cm.sum(1)
        f1 = 2 * tp / denom.clamp_min(1)
        pc = self.pixel_confusion.double()
        inter = pc.diag()
        union = pc.sum(0) + pc.sum(1) - inter
        iou = (inter[union > 0] / union[union > 0]).mean()
        mp, npred, mt, ntarget = self.boundary_totals
        precision = mp / npred if npred else 0.0
        recall = mt / ntarget if ntarget else 0.0
        bf1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
                "miou": float(iou), "bf1": float(bf1)}
