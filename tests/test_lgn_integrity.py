from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lgn import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    HardLogicGateGPTLayer,
    LogicGateGPTLayer,
    ModelConfig,
    TrainConfig,
    _thermometer_ste,
    make_gpt,
)
from pipeline import _make_logic_model, finetune_layers, get_layer_io  # noqa: E402


def test_thermometer_ste_has_unit_total_gradient():
    h = torch.tensor([[0.05, 0.10, 0.12, 0.20, 0.50, 0.90]], requires_grad=True)

    out = _thermometer_ste(h, n_bits=8, training=True)
    out.sum().backward()

    assert out.shape == (1, 48)
    assert torch.equal(out.detach(), out.detach().round())
    assert torch.allclose(h.grad, torch.ones_like(h))


def test_hybrid_freezes_attention_and_keeps_it_eval_after_train():
    torch.manual_seed(0)
    model_cfg = ModelConfig(n_layer=2, n_head=2, n_embd=16, dropout=0.25)
    data_cfg = DataConfig(block_size=8, vocab_size=64)
    baseline, gpt_cfg = make_gpt(model_cfg, data_cfg, device="cpu")

    cfg = ExperimentConfig()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.binary_io = True
    cfg.logic.n_bits = 4
    cfg.logic.sum_pool = True
    cfg.logic.no_in_proj = True

    logic_model = _make_logic_model(baseline, 0, gpt_cfg, cfg.logic, device="cpu")
    layer = logic_model.transformer.h[0]

    assert all(
        not p.requires_grad
        for name, p in layer.named_parameters()
        if name.startswith("attn.") or name.startswith("ln_1.")
    )
    assert any(
        p.requires_grad
        for name, p in layer.named_parameters()
        if name.startswith("logic.") or name.startswith("in_proj.") or name.startswith("out_proj.")
    )

    layer.train()
    assert layer.training
    assert not layer.attn.training
    assert not layer.ln_1.training
    assert layer.ln_2.training


class TinyTokenData:
    def __init__(self, block_size: int = 8, vocab_size: int = 64):
        self.block_size = block_size
        self.vocab_size = vocab_size

    def get_batch(self, split="train", batch_size=2):
        x = torch.randint(0, self.vocab_size, (batch_size, self.block_size))
        y = torch.roll(x, shifts=-1, dims=1)
        return x, y


def test_finetune_resets_ste_toggle():
    torch.manual_seed(1)
    model_cfg = ModelConfig(n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    data_cfg = DataConfig(block_size=8, vocab_size=64)
    baseline, gpt_cfg = make_gpt(model_cfg, data_cfg, device="cpu")

    exp_cfg = ExperimentConfig()
    logic_model = _make_logic_model(baseline, 0, gpt_cfg, exp_cfg.logic, device="cpu")
    layer = logic_model.transformer.h[0]

    train_cfg = TrainConfig()
    train_cfg.batch_size = 2
    train_cfg.finetune_steps = 1
    train_cfg.ste = True
    train_cfg.ft_log_sharpness = False

    finetune_layers(logic_model, [0], TinyTokenData(), train_cfg)

    assert layer.use_ste is False


def test_token_shift_is_channel_aligned_and_soft_hard_match():
    cfg = SimpleNamespace(n_embd=2, dropout=0.0)
    layer = LogicGateGPTLayer(
        cfg,
        layer_idx=0,
        activation="none",
        identity_logic=True,
        binary_io=True,
        n_bits=1,
        sum_pool=True,
        no_in_proj=True,
        token_shift=2,
    )
    layer.norm = nn.Identity()
    layer.eval()

    x = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ]
    )
    y_soft = layer(x)
    y_hard = HardLogicGateGPTLayer(layer)(x)
    contrib = y_soft - x

    expected = torch.tensor(
        [
            [
                [-1.0 / 3.0, -1.0],
                [-1.0 / 3.0, -1.0 / 3.0],
                [1.0 / 3.0, -1.0 / 3.0],
                [-1.0 / 3.0, 1.0 / 3.0],
            ]
        ]
    )

    assert torch.allclose(contrib, expected, atol=1e-6)
    assert torch.allclose(y_soft, y_hard, atol=1e-6)


class AddBlock(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


class MulBlock(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x * self.value


class FakeModel(nn.Module):
    def __init__(self, first_block: nn.Module, second_block: nn.Module):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.transformer = SimpleNamespace(
            drop=nn.Identity(),
            wte=nn.Embedding(8, 3),
            wpe=nn.Embedding(4, 3),
            h=nn.ModuleList([first_block, second_block]),
        )
        nn.init.zeros_(self.transformer.wte.weight)
        nn.init.zeros_(self.transformer.wpe.weight)


class ConstantBatch:
    def get_batch(self, split="train", batch_size=2):
        x = torch.zeros(batch_size, 4, dtype=torch.long)
        y = torch.zeros_like(x)
        return x, y


def test_get_layer_io_uses_live_prefix_but_teacher_target_layer():
    trained_model = FakeModel(AddBlock(1.0), MulBlock(2.0))
    input_model = copy.deepcopy(trained_model)
    input_model.transformer.h[0] = AddBlock(5.0)

    layer_in, layer_tgt = get_layer_io(
        trained_model,
        layer_idx=1,
        data=ConstantBatch(),
        batch_size=2,
        input_model=input_model,
    )

    assert torch.allclose(layer_in, torch.full_like(layer_in, 5.0))
    assert torch.allclose(layer_tgt, torch.full_like(layer_tgt, 10.0))


def test_aggressive_width_mult_is_documented_noop_for_bit_width():
    cfg = SimpleNamespace(n_embd=8, dropout=0.0)
    small = LogicGateGPTLayer(
        cfg,
        layer_idx=0,
        logic_width=8,
        binary_io=True,
        n_bits=4,
        sum_pool=True,
        no_in_proj=True,
    )
    large = LogicGateGPTLayer(
        cfg,
        layer_idx=0,
        logic_width=8 * 16,
        binary_io=True,
        n_bits=4,
        sum_pool=True,
        no_in_proj=True,
    )

    assert small.logic[0].in_dim == large.logic[0].in_dim == 32
    assert small.group_size == large.group_size == 4
