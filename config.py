"""Configuration for NPF-Net and the experimental protocol."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings for a square RGB input."""

    num_classes: int
    image_size: int = 256
    patch_size: int = 4
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout: float = 0.3
    query_chunk: int = 256
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("At least two classes are required.")
        if self.image_size < 32 or self.image_size % 32:
            raise ValueError("Image size must be a positive multiple of 32.")
        if self.patch_size != 4:
            raise ValueError("The architecture requires a patch size of four.")
        if min(self.embed_dim, self.depth, self.num_heads, self.mlp_ratio,
               self.query_chunk) < 1 or self.embed_dim % self.num_heads:
            raise ValueError("Invalid Transformer or attention dimensions.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must lie in [0, 1).")

    @property
    def node_counts(self) -> tuple[int, int, int, int]:
        """Node cardinalities in ResNet stage order, from fine to coarse."""
        return tuple((self.image_size // stride) ** 2 for stride in (4, 8, 16, 32))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """Optimization settings and joint-objective coefficients."""

    epochs: int = 150
    batch_size: int = 16
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    lambda_seg: float = 1.0
    lambda_cls: float = 0.5
    lambda_reg: float = 0.1
    mask_threshold: float = 0.5
    boundary_tolerance: int = 2
    dice_epsilon: float = 1e-6
