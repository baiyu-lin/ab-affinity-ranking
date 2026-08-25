"""AffinityRanker: Plan v4 antigen-conditioned ranking model.

Assembled from the registry blocks in tools/attention.py via build_block().

    P_ab = adapter_antiberty(H_ab)            # [B,L_ab,512] -> [B,L_ab,256]
    P_ag = adapter_esm2(H_ag)                 # [B,L_ag,1280] -> [B,L_ag,256]
    P_ag' = interaction_ag_ab(P_ag, P_ab, kv_mask=ab_mask)   # Q = antigen
    A_pool = attn_pool_256(P_ag', mask=ag_mask)
    B_pool = attn_pool_256(P_ab, mask=ab_mask)
    F = bilinear_gate_256(A_pool, B_pool)
    score = head_v4(F.unsqueeze(1))           # AffinityHead expects [B,L,dim]

Struct arm (config struct_ab_dim/struct_ag_dim > 0): structure-aware
embeddings (e.g. ProstT5/SaProt) fused into the dual-tower features.
Three fusion modes (config struct_fusion):

    residual (default):  pooled vector, linear projection added residually
        B_pool += struct_proj_ab(s_ab)        # s_ab [B,struct_ab_dim]
        A_pool += struct_proj_ag(s_ag)        # s_ag [1|B,struct_ag_dim]

    cross_attn:  the pooled vector is projected to k struct tokens
        (struct_tokens, default 8) and the tower pool attends over them
        (same PreLNCrossAttention block as the ag/ab interaction):
        B_pool = XAttn(Q=B_pool.unsqueeze(1), KV=struct_proj_ab(s_ab).view(B,k,256))

    gated_pre:  per-residue struct tokens fused into the antibody tower
        INPUT (before adapter_ab), so the fused representation is what
        later interacts with the antigen. Antibody-side only
        (struct_ag_dim must be 0). Residues are 1:1 aligned with h_ab.
        h_ab += gate * W(LN(s_ab))            # s_ab [B,L_ab,struct_ab_dim]
        gate is a zero-initialized per-dim vector, so training starts
        exactly at the sequence-only baseline (ReZero-style attribution).
        W is bias-free: zero-filled (missing/padded) rows contribute
        exactly nothing.

Rows with a missing struct embedding arrive as zero vectors (baseline
contribution). The projection/cross-attn/gate layers join the fusion param
group (higher lr, no weight decay).

Ablation flags (config dict):
    interaction:  'cross' | 'none'   (none = skip cross-attn, pool P_ag)
    fusion:       'bilinear' | 'concat'  (concat = Linear(512->256) of [A|B])
    antigen_blind: bool   (drop the antigen tower; score from B_pool alone)
    chain_aware:  bool    (T3.5 arm: add a chain-type embedding (0=VH, 1=VL)
                  to the adapter input; requires chain_ids in forward)

The antigen is batched once per group: pass h_ag with batch size 1 and it is
adapted once, then expanded across the list members after the adapter.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from attention import build_block, PreLNCrossAttention  # noqa: E402

DEFAULT_CONFIG = {
    "blocks": {
        "adapter_ab": "adapter_antiberty",
        "adapter_ag": "adapter_esm2",
        "interaction": "interaction_ag_ab",
        "pool": "attn_pool_256",
        "fusion": "bilinear_gate_256",
        "head": "head_v4",
    },
    "interaction": "cross",      # 'cross' | 'none'
    "fusion": "bilinear",        # 'bilinear' | 'concat'
    "antigen_blind": False,
    "chain_aware": False,        # T3.5 arm; default off = v5 behavior
    "struct_ab_dim": 0,          # >0: struct arm on B_pool (e.g. 1024/1280)
    "struct_ag_dim": 0,          # >0: struct arm on A_pool
    "struct_fusion": "residual", # 'residual' | 'cross_attn' | 'gated_pre'
    "struct_tokens": 8,          # cross_attn: k struct tokens as KV
}


class AffinityRanker(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        cfg["blocks"] = {**DEFAULT_CONFIG["blocks"], **cfg.get("blocks", {})}
        if cfg["interaction"] not in ("cross", "none"):
            raise ValueError("interaction must be 'cross' or 'none'")
        if cfg["fusion"] not in ("bilinear", "concat"):
            raise ValueError("fusion must be 'bilinear' or 'concat'")
        self.config = cfg
        b = cfg["blocks"]

        self.adapter_ab = build_block(b["adapter_ab"])
        self.pool_ab = build_block(b["pool"])
        self.antigen_blind = cfg["antigen_blind"]
        self.interaction_kind = cfg["interaction"]
        self.fusion_kind = cfg["fusion"]
        self.chain_aware = bool(cfg.get("chain_aware", False))
        if self.chain_aware:
            self.chain_embed = nn.Embedding(2, 512)  # 0=VH, 1=VL (AB_DIM)
        self.struct_ab_dim = int(cfg.get("struct_ab_dim", 0))
        self.struct_ag_dim = int(cfg.get("struct_ag_dim", 0))
        self.struct_fusion = cfg.get("struct_fusion", "residual")
        if self.struct_fusion not in ("residual", "cross_attn", "gated_pre"):
            raise ValueError("struct_fusion must be 'residual', 'cross_attn' "
                             "or 'gated_pre'")
        self.struct_tokens = int(cfg.get("struct_tokens", 8))
        if self.struct_ag_dim > 0 and self.antigen_blind:
            raise ValueError("struct_ag_dim arm requires the antigen tower "
                             "(antigen_blind=False)")
        if self.struct_fusion == "gated_pre":
            if self.struct_ab_dim <= 0:
                raise ValueError("gated_pre requires struct_ab_dim > 0")
            if self.struct_ag_dim > 0:
                raise ValueError("gated_pre is antibody-side only "
                                 "(struct_ag_dim must be 0)")
            self.struct_ln_ab = nn.LayerNorm(self.struct_ab_dim)
            self.struct_tokproj_ab = nn.Linear(self.struct_ab_dim, 512,
                                               bias=False)  # -> AB_DIM
            self.struct_gate_ab = nn.Parameter(torch.zeros(512))

        def _build_struct_arm(dim):
            if self.struct_fusion == "residual":
                return nn.Linear(dim, 256), None
            proj = nn.Linear(dim, self.struct_tokens * 256)
            xattn = PreLNCrossAttention(q_dim=256, kv_dim=256, common_dim=256,
                                        num_heads=4, dropout=0.1)
            return proj, xattn

        if self.struct_fusion != "gated_pre":
            if self.struct_ab_dim > 0:
                self.struct_proj_ab, self.struct_xattn_ab = \
                    _build_struct_arm(self.struct_ab_dim)
            if self.struct_ag_dim > 0:
                self.struct_proj_ag, self.struct_xattn_ag = \
                    _build_struct_arm(self.struct_ag_dim)

        if not self.antigen_blind:
            self.adapter_ag = build_block(b["adapter_ag"])
            self.pool_ag = build_block(b["pool"])
            if self.interaction_kind == "cross":
                self.interaction = build_block(b["interaction"])
            if self.fusion_kind == "bilinear":
                self.fusion = build_block(b["fusion"])
            else:
                self.fusion = nn.Linear(512, 256)
        self.head = build_block(b["head"])

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[AffinityRanker] trainable params: {n_train:,} "
              f"(interaction={self.interaction_kind}, fusion={self.fusion_kind}, "
              f"antigen_blind={self.antigen_blind}, chain_aware={self.chain_aware}, "
              f"struct_ab={self.struct_ab_dim}, struct_ag={self.struct_ag_dim}, "
              f"struct_fusion={self.struct_fusion})")

    def _fuse_struct(self, pool, s, proj, xattn):
        """residual: pool + proj(s); cross_attn: pool attends over k struct
        tokens projected from s (residual inside PreLNCrossAttention)."""
        if xattn is None:
            return pool + proj(s)
        tokens = proj(s).view(s.shape[0], self.struct_tokens, 256)
        return xattn(pool.unsqueeze(1), tokens).squeeze(1)

    def forward(self, h_ab, ab_mask, h_ag=None, ag_mask=None, chain_ids=None,
                s_ab=None, s_ag=None):
        """h_ab [B,L_ab,512], ab_mask [B,L_ab]; h_ag [1|B,L_ag,1280].

        chain_ids [B,L_ab] (0=VH, 1=VL) is required when chain_aware=True.
        s_ab / s_ag are required when the corresponding struct arm is
        enabled (zero-fill missing rows): pooled [B,struct_ab_dim] /
        [1|B,struct_ag_dim] for residual/cross_attn, per-residue
        [B,L_ab,struct_ab_dim] for gated_pre.
        Returns scores [B].
        """
        if self.chain_aware:
            if chain_ids is None:
                raise ValueError("chain_ids required when chain_aware=True")
            h_ab = h_ab + self.chain_embed(chain_ids)
        if self.struct_fusion == "gated_pre" and self.struct_ab_dim:
            if s_ab is None:
                raise ValueError("s_ab required when struct_ab_dim > 0")
            # s_ab [B,L_ab,struct_ab_dim], zero-filled where missing/padded
            h_ab = h_ab + self.struct_gate_ab * \
                self.struct_tokproj_ab(self.struct_ln_ab(s_ab))
        P_ab = self.adapter_ab(h_ab)
        B_pool = self.pool_ab(P_ab, ab_mask)
        if self.struct_fusion != "gated_pre" and self.struct_ab_dim:
            if s_ab is None:
                raise ValueError("s_ab required when struct_ab_dim > 0")
            B_pool = self._fuse_struct(B_pool, s_ab, self.struct_proj_ab,
                                       self.struct_xattn_ab)
        if self.antigen_blind:
            F = B_pool
        else:
            if h_ag is None:
                raise ValueError("h_ag required unless antigen_blind=True")
            P_ag = self.adapter_ag(h_ag)
            if P_ag.shape[0] == 1 and P_ab.shape[0] > 1:
                P_ag = P_ag.expand(P_ab.shape[0], -1, -1)
                ag_mask = ag_mask.expand(P_ab.shape[0], -1)
            if self.interaction_kind == "cross":
                P_ag = self.interaction(P_ag, P_ab, kv_mask=ab_mask)
            A_pool = self.pool_ag(P_ag, ag_mask)
            if self.struct_ag_dim:
                if s_ag is None:
                    raise ValueError("s_ag required when struct_ag_dim > 0")
                if s_ag.shape[0] == 1 and A_pool.shape[0] > 1:
                    s_ag = s_ag.expand(A_pool.shape[0], -1)
                A_pool = self._fuse_struct(A_pool, s_ag, self.struct_proj_ag,
                                           self.struct_xattn_ag)
            if self.fusion_kind == "bilinear":
                F = self.fusion(A_pool, B_pool)
            else:
                F = self.fusion(torch.cat([A_pool, B_pool], dim=-1))
        return self.head(F.unsqueeze(1))

    # -- optimizer param groups -------------------------------------------
    def fusion_parameters(self):
        """Gate/fusion/struct-proj params get their own group (higher lr,
        no weight decay)."""
        params = []
        if not self.antigen_blind:
            params += list(self.fusion.parameters())
        if self.struct_fusion == "gated_pre":
            if self.struct_ab_dim:
                params += list(self.struct_ln_ab.parameters())
                params += list(self.struct_tokproj_ab.parameters())
                params += [self.struct_gate_ab]
            return params
        if self.struct_ab_dim:
            params += list(self.struct_proj_ab.parameters())
            if self.struct_fusion == "cross_attn":
                params += list(self.struct_xattn_ab.parameters())
        if self.struct_ag_dim:
            params += list(self.struct_proj_ag.parameters())
            if self.struct_fusion == "cross_attn":
                params += list(self.struct_xattn_ag.parameters())
        return params

    def build_param_groups(self, base_lr=3e-4, fusion_lr=1e-3,
                           weight_decay=0.01):
        fusion_ids = {id(p) for p in self.fusion_parameters()}
        fusion, decay, no_decay = [], [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in fusion_ids:
                fusion.append(p)
            elif p.ndim >= 2:  # weight matrices only
                decay.append(p)
            else:              # biases, norms, scalars
                no_decay.append(p)
        groups = [
            {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        ]
        if fusion:
            groups.append({"params": fusion, "lr": fusion_lr,
                           "weight_decay": 0.0})
        return groups
