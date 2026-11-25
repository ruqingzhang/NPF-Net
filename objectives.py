"""Joint segmentation, classification, and total variation objective."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import TrainConfig
from model import Prediction


class JointLoss(nn.Module):
    """Implement Eqs. (11–15) using numerically stable logits."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, output: Prediction, mask: Tensor, label: Tensor) -> dict[str, Tensor]:
        if mask.shape != output.mask_logits.shape or mask.ndim != 4:
            raise ValueError("Mask and predicted mask must share shape [B, 1, H, W].")
        if label.ndim != 1 or label.shape[0] != mask.shape[0]:
            raise ValueError("Class labels must have shape [B].")
        logits = output.mask_logits.float()
        mask = mask.float()
        prob = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits, mask)
        axes = (1, 2, 3)
        intersection = (prob * mask).sum(dim=axes)
        mass = prob.sum(dim=axes) + mask.sum(dim=axes)
        eps = self.cfg.dice_epsilon
        dice = (1 - (2 * intersection + eps) / (mass + eps)).mean()
        segmentation = bce + dice
        classification = F.cross_entropy(output.class_logits.float(), label.long())
        # Include only valid forward differences; normalize by the full pixel count.
        vertical = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs().sum(dim=axes)
        horizontal = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs().sum(dim=axes)
        tv = ((vertical + horizontal) / (mask.shape[-2] * mask.shape[-1])).mean()
        total = (self.cfg.lambda_seg * segmentation + self.cfg.lambda_cls * classification
                 + self.cfg.lambda_reg * tv)
        return {"loss": total, "segmentation": segmentation, "classification": classification,
                "bce": bce, "dice": dice, "tv": tv}
