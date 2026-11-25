"""Node-guided Pyramid Fusion Network (NPF-Net)."""
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50

from config import ModelConfig


class Prediction(NamedTuple):
    mask_logits: Tensor
    class_logits: Tensor


class PatchEncoder(nn.Module):
    """Encode the spatial patch grid with a pre-normalized ViT."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        self.patch_embed = nn.Conv2d(3, d, cfg.patch_size, cfg.patch_size)
        self.position = nn.Parameter(torch.empty(1, cfg.node_counts[0], d))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg.num_heads, dim_feedforward=d * cfg.mlp_ratio,
                dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(d)

    def forward(self, image: Tensor) -> Tensor:
        tokens = self.patch_embed(image).flatten(2).transpose(1, 2)
        tokens = self.dropout(tokens + self.position)
        for block in self.blocks:
            if self.training and self.cfg.gradient_checkpointing:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return self.norm(tokens)

    def graph_projections(self) -> tuple[Tensor, Tensor]:
        """Share the final ViT attention projections with the token graph."""
        weight = self.blocks[-1].self_attn.in_proj_weight
        q_weight, k_weight, _ = weight.chunk(3, dim=0)
        return q_weight, k_weight


@torch.no_grad()
def incoming_consensus(tokens: Tensor, q_weight: Tensor, k_weight: Tensor,
                       chunk_size: int) -> Tensor:
    """Compute the mean incoming edge weight of each token (Eqs. 3–4)."""
    if tokens.ndim != 3 or tokens.shape[1] == 0 or chunk_size < 1:
        raise ValueError("Expected nonempty tokens [B, N, D] and a positive chunk size.")
    # Use float32 for the graph softmax and accumulation under mixed precision.
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        q = F.linear(tokens.float(), q_weight.float())
        k = F.linear(tokens.float(), k_weight.float()).transpose(1, 2)
        n = tokens.shape[1]
        score = q.new_zeros(q.shape[0], n)
        scale = q.shape[-1] ** -0.5
        for start in range(0, n, chunk_size):
            adjacency = (q[:, start:start + chunk_size] @ k * scale).softmax(dim=-1)
            score += adjacency.sum(dim=1)
        return score / n


def select_nodes(tokens: Tensor, ranking: Tensor, count: int) -> tuple[Tensor, Tensor]:
    """Select a nested Top-K subset and retain the original patch indices."""
    if not 1 <= count <= tokens.shape[1]:
        raise ValueError("Node count exceeds the patch-grid cardinality.")
    indices = ranking[:, :count]
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])), indices


def scatter_pool(nodes: Tensor, indices: Tensor, grid: int, target: int) -> Tensor:
    """Restore patch locations and pool only occupied cells (Eq. 7)."""
    if grid < 1 or target < 1 or grid % target:
        raise ValueError("The target grid must divide the patch grid.")
    b, k, d = nodes.shape
    spatial = nodes.new_zeros(b, grid * grid, d)
    spatial = spatial.scatter(1, indices.unsqueeze(-1).expand(-1, -1, d), nodes)
    occupied = nodes.new_zeros(b, grid * grid, 1)
    occupied = occupied.scatter(1, indices.unsqueeze(-1), nodes.new_ones(b, k, 1))
    window = grid // target
    spatial = spatial.transpose(1, 2).reshape(b, d, grid, grid)
    occupied = occupied.transpose(1, 2).reshape(b, 1, grid, grid)
    numerator = F.avg_pool2d(spatial, window, window)
    denominator = F.avg_pool2d(occupied, window, window)
    return numerator / denominator.clamp_min(1.0 / (window * window))


class StageGuidance(nn.Module):
    """Apply residual cross-attention from CNN positions to selected nodes."""

    def __init__(self, channels: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.q = nn.Linear(channels, cfg.embed_dim, bias=False)
        self.k = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.v = nn.Linear(cfg.embed_dim, channels, bias=False)
        self.chunk_size = cfg.query_chunk
        self.use_checkpoint = cfg.gradient_checkpointing

    @staticmethod
    def _attend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # The single attention distribution and value width follow Eqs. (5–6).
        return F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None],
                                              dropout_p=0.0).squeeze(1)

    def forward(self, feature: Tensor, nodes: Tensor) -> Tensor:
        b, c, h, w = feature.shape
        flat = feature.flatten(2).transpose(1, 2)
        q, k, v = self.q(flat), self.k(nodes), self.v(nodes)
        parts = []
        for query in q.split(self.chunk_size, dim=1):
            if self.training and self.use_checkpoint:
                part = checkpoint(self._attend, query, k, v, use_reentrant=False)
            else:
                part = self._attend(query, k, v)
            parts.append(part)
        delta = torch.cat(parts, dim=1).transpose(1, 2).reshape(b, c, h, w)
        return feature + delta


class PyramidFusion(nn.Module):
    """Propagate coarse semantic features through lateral concatenation."""

    def __init__(self, embed_dim: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.top = nn.Conv2d(channels[-1] + embed_dim, channels[-1], 1)
        self.lateral = nn.ModuleList([
            nn.Conv2d(channels[i] + channels[i + 1], channels[i], 1)
            for i in range(3)
        ])

    def forward(self, features: list[Tensor], spatial_nodes: Tensor) -> Tensor:
        fused = self.top(torch.cat((features[-1], spatial_nodes), dim=1))
        for i in (2, 1, 0):
            up = F.interpolate(fused, size=features[i].shape[-2:], mode="bilinear",
                               align_corners=False)
            fused = self.lateral[i](torch.cat((features[i], up), dim=1))
        return fused


class NPFNet(nn.Module):
    """Joint lesion segmentation and crop disease classification."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(cfg)
        backbone = resnet50(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stages = nn.ModuleList([backbone.layer1, backbone.layer2,
                                     backbone.layer3, backbone.layer4])
        channels = (256, 512, 1024, 2048)
        self.guidance = nn.ModuleList([StageGuidance(c, cfg) for c in channels])
        self.pyramid = PyramidFusion(cfg.embed_dim, channels)
        self.mask_head = nn.Conv2d(channels[0], 1, 1)
        self.class_head = nn.Linear(channels[0], cfg.num_classes)

    def forward(self, image: Tensor) -> Prediction:
        if image.ndim != 4 or image.shape[1:] != (3, self.cfg.image_size, self.cfg.image_size):
            raise ValueError("Input shape must be [B, 3, image_size, image_size].")
        tokens = self.encoder(image)
        q_weight, k_weight = self.encoder.graph_projections()
        scores = incoming_consensus(tokens, q_weight, k_weight, self.cfg.query_chunk)
        ranking = scores.argsort(dim=1, descending=True, stable=True)
        feature = self.stem(image)
        features = []
        for stage, guide in zip(self.stages, self.guidance):
            feature = stage(feature)
            count = feature.shape[-2] * feature.shape[-1]
            nodes, indices = select_nodes(tokens, ranking, count)
            feature = guide(feature, nodes)
            features.append(feature)
        spatial_nodes = scatter_pool(nodes, indices, self.cfg.image_size // self.cfg.patch_size,
                                     features[-1].shape[-1])
        fused = self.pyramid(features, spatial_nodes)
        # Linear mask prediction commutes with bilinear upsampling.
        mask = F.interpolate(self.mask_head(fused), size=image.shape[-2:],
                             mode="bilinear", align_corners=False)
        # Integer-factor bilinear upsampling preserves the spatial mean.
        classification = self.class_head(fused.mean(dim=(2, 3)))
        return Prediction(mask, classification)
