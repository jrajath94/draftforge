"""EAGLE-3 draft head module.

Architecture (per Li et al., NeurIPS'25):
  target hidden states at `layer_indices`
        ↓ (concat)
  fusion_proj: Linear(in_features=L*hidden, out_features=hidden)
        ↓
  decoder_blocks: num_decoder_layers fresh copies of target-layer class
        ↓
  lm_head: tied to target.lm_head (or a copy)

Forward runs the target model with output_hidden_states=True, picks the
features at `layer_indices` (using `+1` because hidden_states[0] is the
embedding output), concatenates along the channel dim, and runs the
draft head.

For training, target features are detached by default — we don't backprop
through Qwen3-4B-Instruct-2507's ~4B parameters, only the head
(~few-hundred-million). This is the standard EAGLE-3 setup (the target is
the teacher; only the draft head is learned).
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class EAGLE3Head(nn.Module):
    """EAGLE-3 draft head with tri-layer feature fusion.

    Args:
        target_model: HF AutoModel (frozen, eval mode, no grad).
        target_config: HF config of the target model (for hidden_size/vocab).
        layer_indices: hidden-state tap indices (must be in [0, num_layers-1]).
        num_decoder_layers: number of fresh decoder blocks. Default 1.
    """

    def __init__(
        self,
        target_model: nn.Module,
        target_config: object,
        layer_indices: list[int],
        num_decoder_layers: int = 1,
    ) -> None:
        super().__init__()
        if len(layer_indices) < 1:
            raise ValueError("layer_indices must be non-empty")

        hidden_size: int = int(target_config.hidden_size)  # type: ignore[attr-defined]
        vocab_size: int = int(target_config.vocab_size)  # type: ignore[attr-defined]
        num_layers: int = int(target_config.num_hidden_layers)  # type: ignore[attr-defined]

        for idx in layer_indices:
            if idx < 0 or idx >= num_layers:
                raise ValueError(
                    f"layer index {idx} out of range [0, {num_layers - 1}]"
                )

        self.target_model = target_model
        self.layer_indices = sorted(layer_indices)
        self.num_decoder_layers = num_decoder_layers
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # 1. Fusion projection — concatenate tap features along channel dim.
        self.fusion_proj = nn.Linear(
            in_features=len(self.layer_indices) * hidden_size,
            out_features=hidden_size,
        )

        # 2. Decoder blocks — fresh random-init copies of one target layer.
        # Take a single target decoder layer, deep-copy it to break the
        # weight tie, then randomly init the copy.
        target_layers = getattr(target_model, "model", None)
        target_layers = getattr(target_layers, "layers", None) if target_layers is not None else None
        if target_layers is None or len(target_layers) == 0:
            raise ValueError(
                "target_model.model.layers not found — expected an HF causal-LM"
            )
        template_layer = target_layers[0]
        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_decoder_layers):
            blk = copy.deepcopy(template_layer)
            # Random-init the copy (decouple from target weights)
            for p in blk.parameters():
                p.requires_grad = True
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)
            self.decoder_blocks.append(blk)

        # 3. LM head — copy from target (do not retrain vocabulary projection).
        target_lm_head = getattr(target_model, "lm_head", None)
        if target_lm_head is None:
            raise ValueError("target_model.lm_head not found")
        self.lm_head = copy.deepcopy(target_lm_head)

        # Freeze target model and target LM head; only head trains.
        for p in self.target_model.parameters():
            p.requires_grad = False
        # The lm_head *copy* is trainable; the target's lm_head isn't.

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
    ) -> torch.Tensor:
        """Run target then draft head. Returns logits over vocabulary.

        Args:
            input_ids: (B, L)
            attention_mask: (B, L) padding mask, or (B, 1, L, L) bool custom
                mask (packed path — HF honors 4-D masks as-is)
            position_ids: (B, L)
            past_key_values: optional HF cache for incremental decoding

        Returns:
            logits: (B, L, V)
        """
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=False,  # training: full forward
            )

        # hidden_states tuple has length (num_layers + 1); [0] = embeddings,
        # [i] = output of layer i-1. We want layer outputs.
        all_hidden = outputs.hidden_states
        if all_hidden is None:
            raise RuntimeError("target_model did not return hidden_states")
        # Silent mis-tap (off-by-one in layer index) would poison an entire
        # $70+ training run; assert expected tuple length before indexing.
        assert len(all_hidden) == self.num_hidden_layers + 1, (
            f"target_model returned {len(all_hidden)} hidden states; "
            f"expected {self.num_hidden_layers + 1} (embedding + {self.num_hidden_layers} layers)"
        )
        # Hidden states index by output layer (1-based offset)
        taps = [all_hidden[i + 1] for i in self.layer_indices]
        fused = torch.cat(taps, dim=-1)  # (B, L, L*hidden)
        h = self.fusion_proj(fused)  # (B, L, hidden)

        # Run fresh decoder blocks (causal; attention mask propagates)
        for blk in self.decoder_blocks:
            out = blk(
                hidden_states=h,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            # HF layer outputs vary across versions; tolerate dict / tuple / Tensor
            if isinstance(out, dict):
                h = out["hidden_states"]
            elif isinstance(out, tuple):
                h = out[0]
            else:
                h = out

        logits = self.lm_head(h)  # (B, L, V)
        return logits

    def num_parameters(self) -> int:
        """Trainable parameter count (for budget / verification)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
