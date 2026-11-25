"""Fixed-partition image and binary lesion-mask loading."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    label: int
    split: str


def read_manifest(path: str | Path, num_classes: int) -> dict[str, list[Sample]]:
    """Load a fixed partition and reject duplicate image or mask paths."""
    path = Path(path).resolve()
    partitions = {"train": [], "val": [], "test": []}
    images, masks = set(), set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "mask", "label", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires image, mask, label, and split columns.")
        for row in reader:
            image = (path.parent / row["image"]).resolve()
            mask = (path.parent / row["mask"]).resolve()
            label, split = int(row["label"]), row["split"].strip().lower()
            if split not in partitions or not 0 <= label < num_classes:
                raise ValueError(f"Invalid split or label: {row}")
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"Missing image or mask: {image}, {mask}")
            if image in images or mask in masks or image == mask:
                raise ValueError(f"Duplicate or overlapping image/mask path: {image}, {mask}")
            images.add(image)
            masks.add(mask)
            partitions[split].append(Sample(image, mask, label, split))
    if images & masks:
        raise ValueError("Image and mask paths must be disjoint.")
    if any(not partition for partition in partitions.values()):
        raise ValueError("Training, validation, and test partitions must be nonempty.")
    return partitions


class CropDiseaseDataset(Dataset):
    """Read paired RGB images and binary masks without modifying the split."""

    def __init__(self, samples: list[Sample], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size
        self.mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        size = (self.image_size, self.image_size)
        with Image.open(sample.image) as source, Image.open(sample.mask) as annotation:
            if source.size != annotation.size:
                raise ValueError(f"Image and mask dimensions differ: {sample.image}")
            # Preserve palette indices rather than converting display colors to luminance.
            raw = np.asarray(annotation)
            if raw.ndim == 3 and raw.shape[2] == 3 and np.all(raw == raw[..., :1]):
                raw = raw[..., 0]
            if raw.ndim != 2:
                raise ValueError(f"Expected a single-channel binary mask: {sample.mask}")
            if not (np.isin(raw, (0, 1)).all() or np.isin(raw, (0, 255)).all()):
                raise ValueError(f"Nonbinary mask: {sample.mask}")
            rgb = np.array(source.convert("RGB").resize(size, Image.Resampling.BILINEAR))
            binary_image = Image.fromarray((raw > 0).astype(np.uint8))
            binary = np.array(binary_image.resize(size, Image.Resampling.NEAREST)) > 0
        image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
        image = (image - self.mean) / self.std
        mask = torch.from_numpy(binary).unsqueeze(0).float()
        return image, mask, torch.tensor(sample.label, dtype=torch.long)
