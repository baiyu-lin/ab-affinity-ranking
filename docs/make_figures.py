#!/usr/bin/env python
"""Generate report figures (docs/assets/).

  fig_architecture.svg/.png  model architecture (frozen dual towers + head)
  fig_dataflow.svg/.png      end-to-end data flow (pipeline -> submission)
  fig_benchmark.png          v3 self-eval: deleak vs old vs chain-aware arm
  fig_loss_curves.png        train loss + monitor Spearman (needs
                             results_deleak_s{0,1,2}.jsonl; skipped if absent)

SVGs are the primary deliverable (vector); PNGs (300 dpi) are for docx
embedding. Run: code/.venv/bin/python docs/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "svg.fonttype": "path",      # text as paths: font-independent SVG
    "figure.dpi": 300,
})

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

C_FROZEN = "#dbeafe"   # frozen towers
C_TRAIN = "#dcfce7"    # trainable blocks
C_DATA = "#fef3c7"     # data artifacts
C_ACT = "#f3e8ff"      # pipeline actions
C_WARN = "#fee2e2"     # de-leakage
C_EDGE = "#475569"


def box(ax, x, y, w, h, text, fc, fs=10.5, ec="#334155", bold_first=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, text=None, fs=9, style="-|>", lw=1.4,
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, lw=lw, color=C_EDGE,
                                 connectionstyle=connectionstyle))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, text, ha="center",
                fontsize=fs, color=C_EDGE)


def elbow(ax, pts, text=None, fs=9, lw=1.4, text_xy=None):
    """Orthogonal polyline with arrowhead on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=C_EDGE, lw=lw, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=C_EDGE))
    if text:
        ax.text(*text_xy, text, ha="center", fontsize=fs, color=C_EDGE,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.9))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(13.2, 10.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- zone bands (gap between bands so labels sit undisturbed) ----------
    ax.add_patch(FancyBboxPatch((0.02, 0.615), 0.96, 0.325,
                                boxstyle="round,pad=0.006", fc="#eff6ff",
                                ec="#bfdbfe", lw=1.2, zorder=0))
    ax.add_patch(FancyBboxPatch((0.02, 0.015), 0.96, 0.565,
                                boxstyle="round,pad=0.006", fc="#f0fdf4",
                                ec="#bbf7d0", lw=1.2, zorder=0))
    ax.text(0.04, 0.918, "冻结区（预训练权重，推理式前向，一次性缓存）",
            fontsize=10.5, color="#1d4ed8", weight="bold", va="center")
    ax.text(0.04, 0.556, "可训练区（1,860,737 参数：适配器 / 交互 / 融合 / 打分头）",
            fontsize=10.5, color="#15803d", weight="bold", va="center")

    ax.text(0.5, 0.978, "抗原条件化亲和力排序模型",
            ha="center", fontsize=16, weight="bold", color="#0f172a")
    ax.text(0.5, 0.952, "双塔冻结 PLM + 交叉注意力 + 双线性门控融合 + ListMLE 排序头",
            ha="center", fontsize=11, color="#64748b")

    # ---- frozen lane --------------------------------------------------------
    box(ax, 0.07, 0.80, 0.30, 0.075, "抗体序列（每组 ≤20 条）\nVH + VL", C_DATA, 10.5)
    box(ax, 0.63, 0.80, 0.30, 0.075, "抗原序列\n（逐行对应，可缺失）", C_DATA, 10.5)
    box(ax, 0.07, 0.655, 0.30, 0.105,
        "抗体塔  AntiBERTy（冻结）\n[CLS] VH [SEP] / [CLS] VL [SEP]\n去特殊符，重→轻链拼接", C_FROZEN, 10)
    box(ax, 0.63, 0.655, 0.30, 0.105,
        "抗原塔  ESM-2 650M（冻结）\n>1000 aa 滑窗 1000/500 重叠平均", C_FROZEN, 10)

    # tensor labels in the inter-band gap, beside the arrows
    _lb = dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=1.0)
    ax.text(0.235, 0.60, "H_ab ∈ R[L_ab × 512]（fp16 缓存）", ha="left",
            fontsize=9, color=C_EDGE, style="italic", bbox=_lb)
    ax.text(0.765, 0.60, "H_ag ∈ R[L_ag × 1280]（fp16 缓存）", ha="left",
            fontsize=9, color=C_EDGE, style="italic", bbox=_lb)

    # ---- trainable: adapters -------------------------------------------------
    box(ax, 0.07, 0.45, 0.30, 0.085,
        "抗体适配器\nLinear → LN → GEGLU-FFN + 残差", C_TRAIN, 10)
    box(ax, 0.63, 0.45, 0.30, 0.085,
        "抗原适配器\nLinear → LN → GEGLU-FFN + 残差", C_TRAIN, 10)
    ax.text(0.235, 0.415, "P_ab ∈ R[L_ab × 256]", ha="left", fontsize=9,
            color=C_EDGE, style="italic", bbox=_lb)
    ax.text(0.765, 0.415, "P_ag ∈ R[L_ag × 256]", ha="left", fontsize=9,
            color=C_EDGE, style="italic", bbox=_lb)

    # ---- row B: pool_ab (left) / cross-attention (right) ---------------------
    box(ax, 0.07, 0.29, 0.26, 0.075,
        "AttnPool(P_ab)\nB_pool ∈ R[256]（抗体摘要）", C_TRAIN, 10)
    box(ax, 0.56, 0.28, 0.37, 0.095,
        "交叉注意力（Pre-LN，4 头，抗原为 Query）\nP_ag′ = P_ag + CrossAttn(Q=LN(P_ag), KV=LN(P_ab))",
        C_TRAIN, 9.5)

    # ---- row C: fusion (left-center) / pool_ag (right) -----------------------
    box(ax, 0.10, 0.155, 0.40, 0.09,
        "双线性门控融合\ngate = σ(W(LN(A)⊙LN(B)) + b)\nF = gate⊙A + (1−gate)⊙B",
        C_TRAIN, 9.5)
    box(ax, 0.60, 0.155, 0.30, 0.075,
        "AttnPool(P_ag′)\nA_pool ∈ R[256]（抗原摘要）", C_TRAIN, 10)

    # ---- row D: head + output -------------------------------------------------
    box(ax, 0.24, 0.04, 0.24, 0.07, "打分头\nMLP 256→64→1（GELU）", C_TRAIN, 10)
    box(ax, 0.60, 0.04, 0.26, 0.07, "score → 组内排序\nRank 1–20", C_DATA, 10)

    # training objective note (bottom-left corner)
    ax.text(0.04, 0.065,
            "训练目标：ListMLE（PL 似然）\n+ 并列扰动 / 截断钉底\n+ rank-only 降权",
            fontsize=8.5, va="center", color="#334155",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#94a3b8"))

    # ---- edges (orthogonal where they cross lanes) ----------------------------
    arrow(ax, 0.22, 0.80, 0.22, 0.763)
    arrow(ax, 0.78, 0.80, 0.78, 0.763)
    arrow(ax, 0.22, 0.655, 0.22, 0.538)
    arrow(ax, 0.78, 0.655, 0.78, 0.538)
    arrow(ax, 0.22, 0.45, 0.22, 0.368)                       # P_ab -> pool_ab
    elbow(ax, [(0.37, 0.492), (0.46, 0.492), (0.46, 0.327), (0.555, 0.327)],
          text="K / V", text_xy=(0.452, 0.44))              # P_ab -> cross (KV)
    arrow(ax, 0.745, 0.45, 0.745, 0.378)                      # P_ag -> cross (Q)
    ax.text(0.762, 0.36, "Q", fontsize=9, color=C_EDGE, style="italic")
    arrow(ax, 0.745, 0.28, 0.745, 0.233)                      # cross -> pool_ag
    arrow(ax, 0.20, 0.29, 0.20, 0.248)                        # pool_ab -> fusion
    ax.text(0.215, 0.262, "B_pool", fontsize=9, color=C_EDGE, style="italic")
    arrow(ax, 0.60, 0.192, 0.505, 0.192)                      # pool_ag -> fusion
    ax.text(0.552, 0.205, "A_pool", fontsize=9, color=C_EDGE, style="italic")
    arrow(ax, 0.36, 0.155, 0.36, 0.113)                       # fusion -> head
    arrow(ax, 0.48, 0.075, 0.595, 0.075)                      # head -> score

    fig.savefig(ASSETS / "fig_architecture.svg", bbox_inches="tight")
    fig.savefig(ASSETS / "fig_architecture.png", bbox_inches="tight")
    plt.close(fig)


def fig_dataflow():
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.965, "端到端数据流：公开语料 → 去泄漏 → 嵌入缓存 → 训练 → 集成推理 → 产出",
            ha="center", fontsize=14.5, weight="bold", color="#0f172a")

    def stage(x, y, num, title):
        ax.add_patch(plt.Circle((x, y), 0.015, fc="#1e40af", ec="none"))
        ax.text(x, y, num, ha="center", va="center", fontsize=9,
                color="white", weight="bold")
        ax.text(x + 0.022, y, title, fontsize=10.5, color="#1e40af",
                weight="bold", va="center")

    # ---- row 1: data pipeline ----------------------------------------------
    stage(0.03, 0.845, "1", "语料清洗与去泄漏")
    box(ax, 0.03, 0.66, 0.155, 0.135, "83 个公开源 CSV\n（AbRank 主体等）", C_DATA, 10)
    box(ax, 0.225, 0.66, 0.155, 0.135, "清洗管线 01–07\n恒定区修剪 / 标签审计\n聚类 / 防泄漏划分", C_ACT, 9.5)
    box(ax, 0.42, 0.66, 0.15, 0.135, "统一语料\n641,454 行\n5,697 抗原组", C_DATA, 10)
    box(ax, 0.61, 0.66, 0.175, 0.135, "v3 基准去泄漏\n40 条精确匹配 + 簇扩展\n剔除 101 行", C_WARN, 9.5)
    box(ax, 0.825, 0.66, 0.145, 0.135, "训练语料\n579,829 合格行\n2,274 合格组", C_DATA, 9.5)
    arrow(ax, 0.185, 0.727, 0.225, 0.727)
    arrow(ax, 0.38, 0.727, 0.42, 0.727)
    arrow(ax, 0.57, 0.727, 0.61, 0.727)
    arrow(ax, 0.785, 0.727, 0.825, 0.727)

    # ---- row 2: embeddings + training ---------------------------------------
    stage(0.24, 0.555, "2", "冻结嵌入与训练")
    box(ax, 0.24, 0.37, 0.21, 0.14, "嵌入提取（一次性，冻结双塔）\nAntiBERTy 抗体 235,331 条 / 48G\nESM-2 650M 抗原 4,716 条 / 4.9G", C_ACT, 9.5)
    box(ax, 0.51, 0.37, 0.17, 0.14, "ListMLE 训练\n3 种子 × 10 轮 × 1500 步\n约 6 分钟/种子", C_ACT, 9.5)
    box(ax, 0.74, 0.37, 0.15, 0.14, "最终检查点\ndeleak_final\ns0 / s1 / s2", C_DATA, 10)
    # feedback: 训练语料 -> 嵌入提取（正交走线，不穿过任何节点）
    elbow(ax, [(0.90, 0.66), (0.90, 0.60), (0.345, 0.60), (0.345, 0.512)],
          text="只删行，缓存直接复用", text_xy=(0.62, 0.612))
    arrow(ax, 0.45, 0.44, 0.51, 0.44)
    arrow(ax, 0.68, 0.44, 0.74, 0.44)

    # ---- row 3: inference + final prediction --------------------------------
    stage(0.03, 0.26, "3", "推理、集成与产出")
    box(ax, 0.03, 0.08, 0.17, 0.135, "v3 评测集\n2 组 × 20 条\n（v3 xlsx 脚本解析）", C_DATA, 9.5)
    box(ax, 0.28, 0.08, 0.18, 0.135, "3 种子 Borda 集成\n组内排序 Rank 1–20\n（每组 1–N 排列断言）", C_ACT, 9.5)
    box(ax, 0.54, 0.08, 0.19, 0.135, "产出组内排序\nfinal_predictions.xlsx", C_DATA, 9.5)
    box(ax, 0.81, 0.08, 0.16, 0.135, "公开真值独立自评\n仅校验，不用于产出\nSpearman 0.667", C_WARN, 9.5)
    # v3 评测集 -> 嵌入提取（正交走线，竖段右移避开行标题）
    elbow(ax, [(0.20, 0.215), (0.20, 0.325), (0.29, 0.325), (0.29, 0.368)])
    # 检查点 -> Borda 集成（正交走线）
    elbow(ax, [(0.815, 0.37), (0.815, 0.29), (0.37, 0.29), (0.37, 0.217)])
    arrow(ax, 0.20, 0.147, 0.28, 0.147)
    arrow(ax, 0.46, 0.147, 0.54, 0.147)
    arrow(ax, 0.73, 0.147, 0.81, 0.147, lw=1.2)

    # legend
    handles = [
        ("数据制品", C_DATA), ("处理动作", C_ACT), ("合规控制（去泄漏/自评边界）", C_WARN),
    ]
    for i, (label, c) in enumerate(handles):
        ax.add_patch(FancyBboxPatch((0.055 + i * 0.16, 0.015), 0.022, 0.026,
                                    boxstyle="round,pad=0.004", fc=c, ec="#334155", lw=0.8))
        ax.text(0.083 + i * 0.16, 0.028, label, fontsize=9, va="center", color="#334155")

    fig.savefig(ASSETS / "fig_dataflow.svg", bbox_inches="tight")
    fig.savefig(ASSETS / "fig_dataflow.png", bbox_inches="tight")
    plt.close(fig)


def fig_benchmark():
    groups = ["Group 1\nSARS-CoV-2", "Group 2\nHER2", "两组均值"]
    deleak = [0.395, 0.938, 0.667]
    old = [0.050, 0.931, 0.490]
    chain = [0.278, 0.786, 0.532]
    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for off, vals, name, c in [(-w, deleak, "去泄漏重训（最终模型）", "#2563eb"),
                               (0, old, "未去泄漏对照臂（同环境）", "#94a3b8"),
                               (w, chain, "链感知消融臂（否决）", "#f59e0b")]:
        bars = ax.bar(x + off, vals, w, label=name, color=c)
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set_xticks(x, groups)
    ax.set_ylabel("与公开真值的 Spearman（3 种子 Borda 集成）")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("v3 基准自评：去泄漏重训 vs 未去泄漏（同环境） vs 消融臂", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_benchmark.png", bbox_inches="tight")
    plt.close(fig)


def fig_loss_curves():
    runs = []
    for s in (0, 1, 2):
        p = ASSETS / f"results_deleak_s{s}.jsonl"
        if not p.exists():
            print(f"skip loss curves: {p} missing (instance offline?)")
            return
        runs.append([json.loads(l) for l in open(p)])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for s, recs in enumerate(runs):
        ep = [r["epoch"] for r in recs]
        a1.plot(ep, [r["train_loss"] for r in recs], marker="o", ms=4,
                label=f"seed {s}")
        a2.plot(ep, [r["val_spearman"] for r in recs], marker="s", ms=4,
                label=f"seed {s}")
        best = max(range(len(recs)), key=lambda i: recs[i]["val_spearman"])
        a2.annotate(f"best {recs[best]['val_spearman']:.3f}@ep{best}",
                    (best, recs[best]["val_spearman"]), fontsize=8,
                    textcoords="offset points", xytext=(4, -12))
    a1.set_xlabel("epoch"); a1.set_ylabel("ListMLE train loss")
    a1.set_title("训练损失收敛（去泄漏重训）")
    a2.set_xlabel("epoch"); a2.set_ylabel("monitor Spearman")
    a2.set_title("监控集 Spearman（仅用于选轮）")
    for a in (a1, a2):
        a.legend(fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_loss_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_loss_curves.png")


if __name__ == "__main__":
    fig_architecture()
    fig_dataflow()
    fig_benchmark()
    fig_loss_curves()
    print("figures ->", ASSETS)
