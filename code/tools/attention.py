"""Attention-level blocks for the Plan v3 affinity-ranking fusion model.

Upper wrappers around the gating/activation units in ``tools/activation.py``.
All hyperparameters come from ``tools/attention_block_params.json``; gating
units are referenced by name from ``tools/gating_activation_params.json``
(unified parameter management).

Architecture spec: preparation/plan-v4.md v3 Part 2,
                   preparation/gating-activation-design.md rev 2
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from activation import build_unit, load_params  # noqa: E402

BLOCK_PARAMS_PATH = Path(__file__).parent / "attention_block_params.json"


def load_block_params(path=None):
    """Load the block parameter registry JSON."""
    with open(path or BLOCK_PARAMS_PATH) as f:
        return json.load(f)


def _build_ref(name, gating_config, dim=None, unit_overrides=None):
    """Build a gating-registry unit by name, overriding dim/fields if given."""
    if (dim is not None or unit_overrides) and "dim" in gating_config["units"][name]:
        import copy
        gating_config = copy.deepcopy(gating_config)
        if dim is not None:
            gating_config["units"][name]["dim"] = dim
        if unit_overrides:
            gating_config["units"][name].update(unit_overrides)
    return build_unit(name, gating_config)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class DualBranchAdapter(nn.Module):
    """Symmetric per-stream adapter (Plan v3): softmax-gated GAU projection.

    out = softmax(W_s . LN(x)) * GAU(LN(x))     x: [B, L, dim] -> [B, L, out_dim]

    Branch 1 (gate): competitive feature selection over out_dim channels.
    Branch 2 (gau): gated content unit (A1 ablation: swiglu/geglu/
    glu_sigmoid/bilinear/gelu_ffn). No residual: this is a projection,
    not a refinement.
    """

    def __init__(self, dim, out_dim, gate, gau, gating_config):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.gate = _build_ref(gate, gating_config, dim=dim)
        self.gau = _build_ref(gau, gating_config, dim=dim)

    def forward(self, x):
        n = self.ln(x)
        return self.gate(n) * self.gau(n)


class GatedCrossAttention(nn.Module):
    """One-shot fusion: the query stream is updated by the KV stream.

    out = Linear_q_dim->common(LN(h_q)) + tanh(a) * CrossAttn(Q=h_q, K/V=h_kv)

    Plan v3: Q = IgLM (512), K/V = ProGen2-small (768). Wired after the
    DualBranchAdapters it operates on the 256-d adapter outputs (the class
    itself is dim-agnostic). Scalar tanh gate init 0 -> output is exactly
    the projected query stream at initialization.
    """

    def __init__(self, q_dim, kv_dim, common_dim, num_heads, dropout,
                 residual_gate, gating_config):
        super().__init__()
        self.ln_q = nn.LayerNorm(q_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)
        self.w_q = nn.Linear(q_dim, common_dim)
        self.w_k = nn.Linear(kv_dim, common_dim)
        self.w_v = nn.Linear(kv_dim, common_dim)
        self.attn = nn.MultiheadAttention(common_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.res_proj = nn.Linear(q_dim, common_dim)
        self.gate = _build_ref(residual_gate, gating_config)

    def forward(self, h_q, h_kv):
        nq = self.ln_q(h_q)
        nkv = self.ln_kv(h_kv)
        c, _ = self.attn(self.w_q(nq), self.w_k(nkv), self.w_v(nkv))
        return self.gate(self.res_proj(nq), c)


class GatedResidualFusion(nn.Module):
    """Cheapest fusion arm (Plan v3 ablation A3): gated linear injection.

    out = W_r(LN(h_q)) + tanh(alpha) * W_p(LN(h_kv))

    Scalar alpha init 0 -> output is exactly the projected query stream
    at initialization.
    """

    def __init__(self, q_dim, kv_dim, common_dim, residual_gate,
                 gating_config):
        super().__init__()
        self.ln_q = nn.LayerNorm(q_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)
        self.res_proj = nn.Linear(q_dim, common_dim)
        self.kv_proj = nn.Linear(kv_dim, common_dim)
        self.gate = _build_ref(residual_gate, gating_config)

    def forward(self, h_q, h_kv):
        return self.gate(self.res_proj(self.ln_q(h_q)),
                         self.kv_proj(self.ln_kv(h_kv)))


class AffinityHead(nn.Module):
    """MLP scoring head: MeanPool -> [+ bypass] -> Linear -> act -> dropout -> scalar.

    bypass = zero-shot perplexity anchors [log ppl_iglm, log ppl_progen],
    concatenated to the pooled vector (plan v3 Part 2).
    """

    def __init__(self, dim, hidden_dim, bypass_dim, dropout, activation,
                 gating_config):
        super().__init__()
        self.bypass_dim = bypass_dim
        self.norm = nn.LayerNorm(dim)  # score-scale invariance
        self.fc1 = nn.Linear(dim + bypass_dim, hidden_dim)
        self.act = _build_ref(activation, gating_config)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h, bypass=None):
        pooled = self.norm(h.mean(dim=1))
        if self.bypass_dim > 0:
            if bypass is None:
                bypass = pooled.new_zeros(pooled.shape[0], self.bypass_dim)
            pooled = torch.cat([pooled, bypass], dim=-1)
        return self.fc2(self.drop(self.act(self.fc1(pooled)))).squeeze(-1)


# ---------------------------------------------------------------------------
# Plan v4 blocks (antigen-conditioned model, update_architecture.txt)
# ---------------------------------------------------------------------------

class ResidualGEGLUAdapter(nn.Module):
    """Tower adapter (Plan v4): x = Linear(dim->out); out = x + Drop(GEGLU_FFN(LN(x))).

    Applied identically to both towers (AntiBERTy 512-d, ESM-2 1280-d -> 256-d).
    GEGLU-FFN internals: Linear(out->hidden) GELU (*) Linear(out->hidden) -> Linear(hidden->out).
    """

    def __init__(self, dim, out_dim, hidden_dim, dropout, gau, gating_config,
                 unit_overrides=None):
        super().__init__()
        overrides = dict(unit_overrides or {})
        overrides.setdefault("hidden_dim", hidden_dim)
        overrides.setdefault("out_dim", out_dim)
        self.proj = nn.Linear(dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.ffn = _build_ref(gau, gating_config, dim=out_dim,
                              unit_overrides=overrides)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        return x + self.drop(self.ffn(self.ln(x)))


class PreLNCrossAttention(nn.Module):
    """Plan v4 interaction: out = x + Drop(MHA(LN(x) as Q, LN(kv) as K/V)).

    Q = antigen tower, K/V = antibody tower (per update_architecture.txt).
    Padding-aware: kv_mask/q_len come from the training harness.
    """

    def __init__(self, q_dim, kv_dim, common_dim, num_heads, dropout):
        super().__init__()
        self.ln_q = nn.LayerNorm(q_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)
        self.w_q = nn.Linear(q_dim, common_dim)
        self.w_k = nn.Linear(kv_dim, common_dim)
        self.w_v = nn.Linear(kv_dim, common_dim)
        self.attn = nn.MultiheadAttention(common_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_q, h_kv, kv_mask=None):
        q = self.w_q(self.ln_q(h_q))
        k = self.w_k(self.ln_kv(h_kv))
        v = self.w_v(self.ln_kv(h_kv))
        kpm = ~kv_mask if kv_mask is not None else None
        c, _ = self.attn(q, k, v, key_padding_mask=kpm)
        return h_q + self.drop(c)


class AttnPool(nn.Module):
    """Additive-attention pooling over the sequence axis (padding-aware)."""

    def __init__(self, dim, attn_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, attn_dim)
        self.w2 = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h, mask=None):
        s = self.w2(torch.tanh(self.w1(h))).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask, -1e4)
        a = torch.softmax(s, dim=1)
        return (h * a.unsqueeze(-1)).sum(dim=1)


class BilinearGateFusion(nn.Module):
    """Plan v4 fusion: gate = sigmoid(W(LN(a) (*) LN(b))); F = gate*a + (1-gate)*b."""

    def __init__(self, dim):
        super().__init__()
        self.ln_a = nn.LayerNorm(dim)
        self.ln_b = nn.LayerNorm(dim)
        self.w = nn.Linear(dim, dim)

    def forward(self, a, b):
        g = torch.sigmoid(self.w(self.ln_a(a) * self.ln_b(b)))
        return g * a + (1 - g) * b


# ---------------------------------------------------------------------------
# Registry-driven factory
# ---------------------------------------------------------------------------

def build_block(name, block_config=None, gating_config=None):
    """Build a block from the registries.

    name:          key in attention_block_params.json["blocks"]
    block_config:  dict from load_block_params() (loaded if omitted)
    gating_config: dict from activation.load_params() (loaded if omitted)
    """
    block_config = block_config or load_block_params()
    gating_config = gating_config or load_params()
    if name not in block_config["blocks"]:
        raise KeyError(f"block '{name}' not in registry")
    p = dict(block_config["blocks"][name])
    btype = p.pop("type")
    p.pop("used_in", None)

    if btype == "dual_branch_adapter":
        return DualBranchAdapter(**p, gating_config=gating_config)
    if btype == "gated_cross_attention":
        return GatedCrossAttention(**p, gating_config=gating_config)
    if btype == "gated_residual_fusion":
        return GatedResidualFusion(**p, gating_config=gating_config)
    if btype == "affinity_head":
        return AffinityHead(**p, gating_config=gating_config)
    if btype == "residual_geglu_adapter":
        return ResidualGEGLUAdapter(**p, gating_config=gating_config)
    if btype == "preln_cross_attention":
        return PreLNCrossAttention(**p)
    if btype == "attn_pool":
        return AttnPool(**p)
    if btype == "bilinear_gate_fusion":
        return BilinearGateFusion(**p)
    raise ValueError(f"unknown block type: {btype}")
