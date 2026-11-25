"""Numerical and integration tests for the NPF-Net implementation."""
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig, TrainConfig
from metrics import DiseaseMetrics, boundary_counts
from model import NPFNet, Prediction, StageGuidance, incoming_consensus, scatter_pool, select_nodes
from objectives import JointLoss


def test_consensus_matches_dense_equations():
    torch.manual_seed(7)
    tokens = torch.randn(2, 13, 12)
    q, k = torch.randn(12, 12), torch.randn(12, 12)
    dense = (F.linear(tokens, q) @ F.linear(tokens, k).transpose(-1, -2) / 12**0.5).softmax(-1)
    expected = dense.mean(1)
    actual = incoming_consensus(tokens, q, k, 5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.sum(1), torch.ones(2))
    rank = actual.argsort(1, descending=True)
    nodes, index = select_nodes(tokens, rank, 4)
    torch.testing.assert_close(nodes, tokens.gather(1, index[..., None].expand(-1, -1, 12)))
    assert torch.equal(index, rank[:, :4])


def test_spatial_reconstruction_and_gradients():
    nodes = torch.tensor([[[2.0], [6.0], [8.0]]], requires_grad=True)
    indices = torch.tensor([[0, 5, 15]])
    pooled = scatter_pool(nodes, indices, 4, 2)
    torch.testing.assert_close(pooled, torch.tensor([[[[4., 0.], [0., 8.]]]]))
    pooled.sum().backward()
    torch.testing.assert_close(nodes.grad, torch.tensor([[[0.5], [0.5], [1.0]]]))
    with pytest.raises(ValueError):
        scatter_pool(nodes, indices, 4, 0)


def test_chunked_cross_attention_matches_dense():
    cfg = ModelConfig(num_classes=3, image_size=32, embed_dim=12, depth=1,
                      num_heads=3, query_chunk=3, gradient_checkpointing=False)
    module = StageGuidance(5, cfg).eval()
    feature = torch.randn(2, 5, 2, 4)
    nodes = torch.randn(2, 7, 12)
    flat = feature.flatten(2).transpose(1, 2)
    weights = (module.q(flat) @ module.k(nodes).transpose(-1, -2) / 12**0.5).softmax(-1)
    expected = feature + (weights @ module.v(nodes)).transpose(1, 2).reshape_as(feature)
    torch.testing.assert_close(module(feature, nodes), expected)


def test_joint_loss_equations_and_gradients():
    cfg = TrainConfig()
    logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
    classes = torch.zeros(2, 3, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0] = 1
    losses = JointLoss(cfg)(Prediction(logits, classes), target, torch.tensor([0, 1]))
    expected = np.log(2) + (1 / 3 + 1) / 2 + 0.5 * np.log(3)
    assert losses["loss"].item() == pytest.approx(expected, abs=1e-6)
    assert losses["tv"].item() == 0
    losses["loss"].backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(classes.grad).all()


def test_metrics_perfect_and_shifted_masks():
    masks = torch.zeros(3, 1, 12, 12)
    masks[:2, :, 3:7, 3:7] = 1
    labels = torch.arange(3)
    output = Prediction(masks * 20 - 10, torch.eye(3) * 20)
    metrics = DiseaseMetrics(3)
    metrics.update(output, masks, labels)
    assert metrics.compute() == dict(accuracy=1., macro_f1=1., miou=1., bf1=1.)
    p = masks[0, 0].numpy().astype(bool)
    shifted = np.roll(p, 3, axis=1)
    mp, npred, mt, ntarget = boundary_counts(p, shifted, 2)
    assert 0 < mp < npred and 0 < mt < ntarget


def test_default_protocol():
    cfg = ModelConfig(num_classes=14)
    assert cfg.node_counts == (4096, 1024, 256, 64)
    assert (cfg.image_size, cfg.patch_size, cfg.embed_dim, cfg.depth) == (256, 4, 768, 12)
    t = TrainConfig()
    assert (t.epochs, t.batch_size, t.learning_rate, t.min_learning_rate) == (150, 16, 1e-4, 1e-6)
    assert (t.lambda_seg, t.lambda_cls, t.lambda_reg) == (1.0, 0.5, 0.1)


def test_model_forward_backward_and_checkpoint(tmp_path):
    torch.manual_seed(11)
    cfg = ModelConfig(num_classes=3, image_size=64, embed_dim=12, depth=1,
                      num_heads=3, query_chunk=64, dropout=0.0)
    net = NPFNet(cfg)
    x = torch.randn(2, 3, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.7).float()
    labels = torch.tensor([1, 2])
    output = net(x)
    assert output.mask_logits.shape == (2, 1, 64, 64)
    assert output.class_logits.shape == (2, 3)
    loss = JointLoss(TrainConfig())(output, target, labels)["loss"]
    loss.backward()
    gradients = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert net.encoder.blocks[-1].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert all(stage.q.weight.grad.abs().sum() > 0 for stage in net.guidance)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    old = net.class_head.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(old, net.class_head.weight)
    net.eval()
    with torch.inference_mode():
        expected = net(x)
    path = tmp_path / "model.pt"
    torch.save({"model": net.state_dict(), "model_config": cfg.to_dict()}, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored = NPFNet(ModelConfig(**state["model_config"]))
    restored.load_state_dict(state["model"])
    restored.eval()
    with torch.inference_mode():
        actual = restored(x)
    torch.testing.assert_close(actual.mask_logits, expected.mask_logits)
    torch.testing.assert_close(actual.class_logits, expected.class_logits)


def test_manifest_and_palette_mask(tmp_path):
    from PIL import Image
    from data import CropDiseaseDataset, read_manifest

    rows = ["image,mask,label,split"]
    for i, split in enumerate(("train", "val", "test")):
        Image.new("RGB", (8, 8), (30, 70, 90)).save(tmp_path / f"image{i}.png")
        mask = Image.fromarray(np.pad(np.ones((4, 4), dtype=np.uint8), 2)).convert("P")
        palette = [0, 0, 0, 255, 0, 0] + [0] * 762
        mask.putpalette(palette)
        mask.save(tmp_path / f"mask{i}.png")
        rows.append(f"image{i}.png,mask{i}.png,{i},{split}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows), encoding="utf-8")
    partitions = read_manifest(manifest, 3)
    image, mask, label = CropDiseaseDataset(partitions["train"], 32)[0]
    assert image.shape == (3, 32, 32) and mask.shape == (1, 32, 32)
    assert mask.sum().item() == 16 * 16 and label.item() == 0
    manifest.write_text("\n".join(rows + ["image0.png,mask0.png,0,test"]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest, 3)


def test_metric_shape_validation_is_atomic():
    metrics = DiseaseMetrics(3)
    output = Prediction(torch.zeros(2, 1, 4, 4), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="shape"):
        metrics.update(output, torch.zeros(2, 1, 1, 4), torch.tensor([0, 1]))
    assert metrics.confusion.sum() == 0
    assert metrics.pixel_confusion.sum() == 0


def test_sample_weighted_accumulation():
    from train import make_scaler, train_epoch
    from torch.utils.data import DataLoader, TensorDataset

    class TinyJointModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mask = nn.Conv2d(3, 1, 1)
            self.cls = nn.Linear(3, 3)

        def forward(self, image):
            return Prediction(self.mask(image), self.cls(image.mean((2, 3))))

    torch.manual_seed(17)
    data = TensorDataset(torch.randn(5, 3, 4, 4),
                         (torch.rand(5, 1, 4, 4) > 0.5).float(), torch.tensor([0, 1, 2, 1, 0]))
    a, b = TinyJointModel(), TinyJointModel()
    b.load_state_dict(a.state_dict())
    criterion = JointLoss(TrainConfig())
    for model, batch, steps in ((a, 5, 1), (b, 2, 3)):
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        train_epoch(model, DataLoader(data, batch_size=batch), criterion, opt,
                    make_scaler(False), torch.device("cpu"), False, steps)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb)


def test_training_cli_and_held_out_evaluation(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from PIL import Image

    rows = ["image,mask,label,split"]
    for i, split in enumerate(("train", "train", "train", "val", "val", "test", "test")):
        label = i % 2
        rgb = np.full((64, 64, 3), 60 + 20 * i, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        if label:
            mask[12:30, 10:29] = 255
            rgb[12:30, 10:29, 0] = 240
        Image.fromarray(rgb).save(tmp_path / f"x{i}.png")
        Image.fromarray(mask).save(tmp_path / f"y{i}.png")
        rows.append(f"x{i}.png,y{i}.png,{label},{split}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows), encoding="utf-8")
    output = tmp_path / "experiment"
    command = [sys.executable, str(Path(__file__).with_name("train.py")), "--manifest", str(manifest),
               "--output", str(output), "--num-classes", "2", "--seeds", "7", "8", "--epochs", "1",
               "--batch-size", "2", "--accumulation-steps", "2", "--image-size", "64",
               "--embed-dim", "12", "--num-heads", "3", "--depth", "1", "--query-chunk", "64",
               "--device", "cpu"]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    for seed in (7, 8):
        folder = output / f"seed_{seed}"
        config = json.loads((folder / "config.json").read_text())
        history = json.loads((folder / "history.json").read_text())
        assert config["partition_sizes"] == {"train": 3, "val": 2, "test": 2}
        assert len(history) == 1 and "test" not in history[0]
        assert (folder / "best.pt").is_file()
        scores = json.loads((folder / "test_metrics.json").read_text())
        assert all(0 <= scores[name] <= 1 for name in ("accuracy", "macro_f1", "miou", "bf1"))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["seeds"] == [7, 8] and len(summary["runs"]) == 2
    assert summary["metrics"]["miou"]["std"] is not None
