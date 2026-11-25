"""Train and evaluate NPF-Net with a fixed held-out test partition."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from data import CropDiseaseDataset, read_manifest
from metrics import DiseaseMetrics
from model import NPFNet
from objectives import JointLoss


def seed_everything(seed: int) -> None:
    """Set the random state for a single independent training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_scaler(enabled: bool):
    """Support both PyTorch 2.1 and later AMP interfaces."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(model, loader, criterion, optimizer, scaler, device, amp,
                accumulation_steps: int = 1) -> dict[str, float]:
    """Optimize sample-weighted batches, including incomplete accumulation groups."""
    if accumulation_steps < 1:
        raise ValueError("Accumulation steps must be positive.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {}
    seen = group_size = 0
    for step, (image, mask, label) in enumerate(loader, start=1):
        image, mask, label = (v.to(device, non_blocking=True) for v in (image, mask, label))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            losses = criterion(model(image), mask, label)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"Nonfinite training objective at step {step}.")
        n = image.size(0)
        scaler.scale(losses["loss"] * n).backward()
        seen += n
        group_size += n
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + value.detach().item() * n
        if step % accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(group_size)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_size = 0
    if not seen:
        raise ValueError("Training loader is empty.")
    return {name: value / seen for name, value in totals.items()}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp, num_classes, cfg) -> dict[str, float]:
    """Evaluate the complete partition without updating model parameters."""
    model.eval()
    metrics = DiseaseMetrics(num_classes, cfg.mask_threshold, cfg.boundary_tolerance)
    total_loss = 0.0
    seen = 0
    for image, mask, label in loader:
        image, mask, label = (v.to(device, non_blocking=True) for v in (image, mask, label))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(image)
            loss = criterion(output, mask, label)["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation objective.")
        total_loss += loss.item() * image.size(0)
        seen += image.size(0)
        metrics.update(output, mask, label)
    return {**metrics.compute(), "loss": total_loss / seen}


def run_seed(seed, partitions, model_cfg, train_cfg, args, device) -> dict:
    seed_everything(seed)
    directory = args.output / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    amp = device.type == "cuda" and not args.no_amp
    rng = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(CropDiseaseDataset(rows, model_cfg.image_size),
                          batch_size=train_cfg.batch_size, shuffle=split == "train",
                          generator=rng if split == "train" else None,
                          num_workers=args.workers, pin_memory=device.type == "cuda")
        for split, rows in partitions.items()
    }
    model = NPFNet(model_cfg).to(device)
    criterion = JointLoss(train_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate,
                                 betas=(train_cfg.beta1, train_cfg.beta2),
                                 weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.epochs, eta_min=train_cfg.min_learning_rate)
    scaler = make_scaler(amp)
    write_json(directory / "config.json", {
        "model": model_cfg.to_dict(), "training": asdict(train_cfg), "seed": seed,
        "partition_sizes": {key: len(value) for key, value in partitions.items()},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "device": str(device), "amp": amp, "accumulation_steps": args.accumulation_steps,
        "torch_version": str(torch.__version__),
    })
    best = float("-inf")
    history = []
    for epoch in range(1, train_cfg.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_epoch(model, loaders["train"], criterion, optimizer, scaler,
                                    device, amp, args.accumulation_steps)
        val_metrics = evaluate(model, loaders["val"], criterion, device, amp,
                               model_cfg.num_classes, train_cfg)
        if val_metrics["miou"] > best:
            best = val_metrics["miou"]
            torch.save({"model": model.state_dict(), "model_config": model_cfg.to_dict(),
                        "epoch": epoch, "validation": val_metrics}, directory / "best.pt")
        history.append({"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics})
        write_json(directory / "history.json", history)
        print(f"Seed {seed} | Epoch {epoch:03d} | Loss {train_metrics['loss']:.4f} | "
              f"Validation mIoU {val_metrics['miou']:.4f}", flush=True)
        scheduler.step()
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    results = evaluate(model, loaders["test"], criterion, device, amp,
                       model_cfg.num_classes, train_cfg)
    write_json(directory / "test_metrics.json", results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--no-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.accumulation_steps) < 1 or args.workers < 0:
        raise ValueError("Invalid training schedule or worker count.")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        raise ValueError("Seeds must be unique unsigned 32-bit integers.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model_cfg = ModelConfig(num_classes=args.num_classes, image_size=args.image_size,
                            embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                            query_chunk=args.query_chunk, gradient_checkpointing=not args.no_checkpointing)
    train_cfg = replace(TrainConfig(), epochs=args.epochs, batch_size=args.batch_size)
    partitions = read_manifest(args.manifest, args.num_classes)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Select a CPU or CUDA device.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment.")
    if model_cfg.image_size == 32 and (train_cfg.batch_size == 1 or
                                      len(partitions["train"]) % train_cfg.batch_size == 1):
        raise ValueError("At 32-pixel resolution, use training batches of at least two samples "
                         "or increase the image size to 64 pixels for batch normalization.")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Output directory must be empty for a new experiment.")
    results = [run_seed(seed, partitions, model_cfg, train_cfg, args, device) for seed in args.seeds]
    summary = {key: {"mean": statistics.mean(r[key] for r in results),
                     "std": statistics.stdev(r[key] for r in results) if len(results) > 1 else None}
               for key in results[0]}
    write_json(args.output / "summary.json", {"seeds": args.seeds, "metrics": summary, "runs": results})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
