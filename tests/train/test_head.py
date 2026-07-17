"""Tests for train/head.py EAGLE3Head.

Uses a stub HF-like target model (no torch dependency for instantiation;
pure nn.Module shape). Real Qwen3-4B-Instruct-2507 is exercised by the integration
training run on rented GPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from train.head import EAGLE3Head


class _StubConfig:
    """Mimics HF PretrainedConfig surface used by EAGLE3Head."""

    hidden_size = 64
    vocab_size = 200
    num_hidden_layers = 4


class _StubLayer(nn.Module):
    """Mimics one HF decoder layer — accepts hidden_states kwarg, returns (hidden_states,)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **_kw):
        # Trivial transformation so logits are non-trivial
        h = hidden_states + self.mlp(hidden_states) * 0.01
        return (h,)


class _StubInnerModel(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_StubConfig.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            _StubLayer(hidden_size) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, **_kw):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(hidden_states=h)[0]
        h = self.norm(h)
        # Mimic HF BaseModelOutputWithPast
        out = type("O", (), {})()
        out.last_hidden_state = h
        out.hidden_states = (h, *tuple(h for _ in self.layers))
        return out


class _StubTargetModel(nn.Module):
    """Mimics HF CausalLM: nested `model` + `lm_head`."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        self.config = _StubConfig()
        self.model = _StubInnerModel(hidden_size=hidden_size, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, _StubConfig.vocab_size, bias=False)

    def forward(self, input_ids, output_hidden_states=False, **_kw):
        # Our stub inner model already returns hidden_states if asked.
        result = self.model(input_ids=input_ids)
        last = result.last_hidden_state
        if not output_hidden_states:
            result.hidden_states = None
        # Build a HF-like output object
        out = type("O", (), {})()
        out.hidden_states = result.hidden_states
        out.last_hidden_state = last
        return out


@pytest.fixture
def stub_target() -> tuple[_StubTargetModel, _StubConfig]:
    torch.manual_seed(0)
    return _StubTargetModel(), _StubConfig()


def test_head_initializes(stub_target) -> None:
    model, cfg = stub_target
    head = EAGLE3Head(
        target_model=model,
        target_config=cfg,
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    assert head.hidden_size == 64
    assert head.layer_indices == [1, 2, 3]
    assert len(head.decoder_blocks) == 1


def test_head_forward_shape(stub_target) -> None:
    model, cfg = stub_target
    head = EAGLE3Head(
        target_model=model,
        target_config=cfg,
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    head.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        logits = head(input_ids=input_ids)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_target_model_frozen(stub_target) -> None:
    """Target params must be detached (requires_grad=False)."""
    model, cfg = stub_target
    head = EAGLE3Head(
        target_model=model,
        target_config=cfg,
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    for p in head.target_model.parameters():
        assert p.requires_grad is False


def test_out_of_range_layer_index_rejected(stub_target) -> None:
    model, cfg = stub_target
    with pytest.raises(ValueError, match=r"out of range"):
        EAGLE3Head(
            target_model=model,
            target_config=cfg,
            layer_indices=[0, 99],
            num_decoder_layers=1,
        )


def test_empty_layer_indices_rejected(stub_target) -> None:
    model, cfg = stub_target
    with pytest.raises(ValueError, match=r"non-empty"):
        EAGLE3Head(
            target_model=model,
            target_config=cfg,
            layer_indices=[],
            num_decoder_layers=1,
        )


def test_backward_updates_trainable_params(stub_target) -> None:
    """At least one head param must accumulate a nonzero gradient after backward."""
    model, cfg = stub_target
    head = EAGLE3Head(
        target_model=model,
        target_config=cfg,
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    head.train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = head(input_ids=input_ids)
    logits.sum().backward()
    grads = [
        p.grad.norm().item()
        for p in head.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert any(g > 0 for g in grads), "no head gradients computed"


def test_fusion_proj_shape(stub_target) -> None:
    model, cfg = stub_target
    head = EAGLE3Head(
        target_model=model,
        target_config=cfg,
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    assert head.fusion_proj.in_features == 3 * cfg.hidden_size
    assert head.fusion_proj.out_features == cfg.hidden_size


def test_decoder_blocks_count_configurable(stub_target) -> None:
    model, cfg = stub_target
    head = EAGLE3Head(
        target_model=model,
        target_config=cfg,
        layer_indices=[1, 2, 3],
        num_decoder_layers=3,
    )
    assert len(head.decoder_blocks) == 3


def test_decoder_blocks_decoupled_from_target(stub_target) -> None:
    """Decoder block tensors must NOT share memory with target layer weights."""
    model, cfg = stub_target
    head = EAGLE3Head(
        target_model=model,
        target_config=cfg,
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    target_q = model.model.layers[0].q_proj.weight
    head_q = head.decoder_blocks[0].q_proj.weight
    assert target_q.data_ptr() != head_q.data_ptr()
    assert not torch.allclose(target_q, head_q)


def test_num_parameters_positive(stub_target) -> None:
    model, cfg = stub_target
    head = EAGLE3Head(
        target_model=model,
        target_config=cfg,
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    assert head.num_parameters() > 0


def test_unsorted_layer_indices_sorted(stub_target) -> None:
    """Constructor must sort layer_indices (later layers first if user passes reversed)."""
    model, cfg = stub_target
    head = EAGLE3Head(
        target_model=model,
        target_config=cfg,
        layer_indices=[3, 1, 2],
        num_decoder_layers=1,
    )
    assert head.layer_indices == [1, 2, 3]


class _BadHiddenCountInnerModel(nn.Module):
    """Stub inner model that returns the WRONG number of hidden states.

    Mimics the failure mode of a mis-tapped target (e.g., older transformers
    versions where output_hidden_states returned N instead of N+1). The
    EAGLE-3 head must refuse rather than silently index into wrong slots
    and poison a $70+ training run.
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 4, bad_count: int = 1) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_StubConfig.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            _StubLayer(hidden_size) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self._bad_count = bad_count

    def forward(self, input_ids, **_kw):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(hidden_states=h)[0]
        h = self.norm(h)
        out = type("O", (), {})()
        out.last_hidden_state = h
        # Truncated hidden_states tuple — too short for the head's taps.
        out.hidden_states = tuple(h for _ in range(self._bad_count))
        return out


class _BadHiddenCountTargetModel(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 4, bad_count: int = 1) -> None:
        super().__init__()
        self.config = _StubConfig()
        self.model = _BadHiddenCountInnerModel(
            hidden_size=hidden_size, num_layers=num_layers, bad_count=bad_count
        )
        self.lm_head = nn.Linear(hidden_size, _StubConfig.vocab_size, bias=False)

    def forward(self, input_ids, output_hidden_states=False, **_kw):
        result = self.model(input_ids=input_ids)
        last = result.last_hidden_state
        if not output_hidden_states:
            result.hidden_states = None
        out = type("O", (), {})()
        out.hidden_states = result.hidden_states
        out.last_hidden_state = last
        return out


def test_off_by_one_hidden_states_rejected() -> None:
    """Silent mis-tap guard: target returning wrong hidden-state tuple length must fail loudly.

    For num_hidden_layers=4 the head expects 5 hidden states (embedding + 4 layers).
    A target returning only 1 must raise AssertionError with a clear message — not
    silently index into wrong positions and poison training.
    """
    bad_target = _BadHiddenCountTargetModel(num_layers=4, bad_count=1)
    head = EAGLE3Head(
        target_model=bad_target,
        target_config=_StubConfig(),
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    head.eval()
    input_ids = torch.randint(0, _StubConfig.vocab_size, (2, 8))
    with pytest.raises(AssertionError, match=r"hidden states"):
        with torch.no_grad():
            head(input_ids=input_ids)


def test_off_by_one_too_many_hidden_states_rejected() -> None:
    """Inverse case: too many hidden states must also fail (catches double-counting)."""
    bad_target = _BadHiddenCountTargetModel(num_layers=4, bad_count=10)
    head = EAGLE3Head(
        target_model=bad_target,
        target_config=_StubConfig(),
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    head.eval()
    input_ids = torch.randint(0, _StubConfig.vocab_size, (2, 8))
    with pytest.raises(AssertionError, match=r"hidden states"):
        with torch.no_grad():
            head(input_ids=input_ids)


def test_no_hidden_states_raises_runtime_error() -> None:
    """target_model that didn't return hidden_states at all must raise RuntimeError."""
    bad_target = _BadHiddenCountTargetModel(num_layers=4, bad_count=1)
    # Override forward to suppress hidden_states entirely
    def _no_hidden(self_, input_ids, output_hidden_states=False, **_kw):
        out = type("O", (), {})()
        out.hidden_states = None
        out.last_hidden_state = self_.model(input_ids=input_ids).last_hidden_state
        return out
    bad_target.forward = _no_hidden.__get__(bad_target)  # type: ignore[method-assign]
    head = EAGLE3Head(
        target_model=bad_target,
        target_config=_StubConfig(),
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    head.eval()
    input_ids = torch.randint(0, _StubConfig.vocab_size, (2, 8))
    with pytest.raises(RuntimeError, match=r"hidden_states"):
        with torch.no_grad():
            head(input_ids=input_ids)
