"""Gating & activation units for the Plan v3 affinity-ranking fusion model.

All units are registry-driven: hyperparameters live in
``tools/gating_activation_params.json`` and units are built via
``build_unit(name, config)``.

Block wiring:       preparation/gating-activation-design.md
General provisions: preparation/data-pipeline-research.md
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PARAMS_PATH = Path(__file__).parent / "gating_activation_params.json"


def load_params(path=None):
    """Load the unit parameter registry JSON."""
    with open(path or PARAMS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Residual-branch gates:  forward(x, branch_out) -> gated residual
# ---------------------------------------------------------------------------

class TanhGate(nn.Module):
    """Flamingo-style tanh gate: x + tanh(alpha) * branch_out.

    alpha init 0 -> exact identity at initialization, bounded to (-1, 1).
    gate_shape: "scalar" or "per_channel" (requires dim).
    """

    def __init__(self, gate_shape="scalar", dim=None, init=0.0):
        super().__init__()
        if gate_shape == "scalar":
            self.alpha = nn.Parameter(torch.full((1,), float(init)))
        elif gate_shape == "per_channel":
            if dim is None:
                raise ValueError("per_channel gate requires dim")
            self.alpha = nn.Parameter(torch.full((dim,), float(init)))
        else:
            raise ValueError(f"unknown gate_shape: {gate_shape}")

    def forward(self, x, branch_out):
        return x + torch.tanh(self.alpha) * branch_out


# ---------------------------------------------------------------------------
# Projection gates (DualBranchAdapter branch 1)
# ---------------------------------------------------------------------------

class ProjectionGate(nn.Module):
    """Feature-selection gate: softmax(W(x)) or sigmoid(W(x)) over channels.

    softmax gives non-negative, sum-to-1 competitive selection; sigmoid is
    the independent-channel ablation arm (A2).
    """

    def __init__(self, dim, out_dim, gate="softmax"):
        super().__init__()
        if gate not in ("softmax", "sigmoid"):
            raise ValueError(f"unknown gate: {gate}")
        self.gate = gate
        self.w = nn.Linear(dim, out_dim)

    def forward(self, x):
        logits = self.w(x)
        if self.gate == "softmax":
            return torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Gated FFNs (GLU family), plain FFNs, and plain activations
# ---------------------------------------------------------------------------

_GLU_VARIANTS = {
    "geglu": F.gelu,
    "swiglu": F.silu,
    "glu": torch.sigmoid,
    "reglu": F.relu,
    "bilinear": lambda t: t,
}


class GatedFFN(nn.Module):
    """GLU-family FFN: out = (act(x @ W_g) * (x @ W_v)) @ W_o.

    variant: geglu | swiglu | glu | reglu | bilinear (Shazeer 2020).
    out_dim defaults to dim; the DualBranchAdapter sets it to the common
    width 256 (ablation A1).
    """

    def __init__(self, dim, hidden_dim, out_dim=None, variant="geglu"):
        super().__init__()
        if variant not in _GLU_VARIANTS:
            raise ValueError(f"unknown GLU variant: {variant}")
        self.variant = variant
        self.w_g = nn.Linear(dim, hidden_dim)
        self.w_v = nn.Linear(dim, hidden_dim)
        self.w_o = nn.Linear(hidden_dim, out_dim if out_dim is not None else dim)

    def forward(self, x):
        gate = _GLU_VARIANTS[self.variant](self.w_g(x))
        return self.w_o(gate * self.w_v(x))


_ACTIVATIONS = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}


class PlainFFN(nn.Module):
    """Ungated FFN: Linear(dim -> hidden) -> activation -> Linear(hidden -> out_dim).

    A1 no-gate control arm for the DualBranchAdapter GAU branch.
    """

    def __init__(self, dim, hidden_dim, out_dim=None, activation="gelu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation}")
        self.w_i = nn.Linear(dim, hidden_dim)
        self.act = _ACTIVATIONS[activation]()
        self.w_o = nn.Linear(hidden_dim, out_dim if out_dim is not None else dim)

    def forward(self, x):
        return self.w_o(self.act(self.w_i(x)))


# ---------------------------------------------------------------------------
# Registry-driven factory
# ---------------------------------------------------------------------------

def build_unit(name, config):
    """Build a unit from the registry.

    name:   key in config["units"] (e.g. "swiglu", "tanh_gate_scalar")
    config: dict from load_params()
    """
    if name not in config["units"]:
        raise KeyError(f"unit '{name}' not in registry")
    p = dict(config["units"][name])
    utype = p.pop("type")
    for meta_key in ("used_in", "reference"):
        p.pop(meta_key, None)

    if utype == "tanh_gate":
        return TanhGate(**p)
    if utype == "projection_gate":
        return ProjectionGate(**p)
    if utype == "gated_ffn":
        return GatedFFN(**p)
    if utype == "plain_ffn":
        return PlainFFN(**p)
    if utype == "activation":
        return _ACTIVATIONS[p["name"]]()
    raise ValueError(f"unknown unit type: {utype}")
