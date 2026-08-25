#!/usr/bin/env python
"""Assemble report.md from the two canonical part documents.

  算法设计文档.md  -> 第一部分（含算法综述与参考文献）
  算法训练文档.md  -> 第二部分

Each part keeps its own internal section numbering; cross-references
between parts use the "训练文档 §x.y" / "设计文档 §x.y" convention.

After assembly, add_crossrefs() post-processes the merged markdown:
  1. Renumbers bibliography entries by order of first in-text citation
     (GB/T 7714 顺序编码制) and rewrites all in-text [n] citations.
  2. Adds span anchors to figure/table caption labels and bibliography
     entry labels, and converts in-text 图 N / 表 N / [n] / §x.y mentions
     into pandoc internal links targeting those anchors.
  3. Appends explicit {#secD_*} / {#secT_*} identifiers to referenced
     section headings.
make_crossrefs.py then converts the fig/tab/ref links in the docx into
real Word REF fields (交叉引用域); § links stay plain internal hyperlinks.

Run: code/.venv/bin/python docs/build_report.py
"""
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent

HEADER = """# 技术报告 —— 抗原-抗体亲和力排序

**日期**：2026-08-12

## 报告与发布包总览

### 文档组织

本报告由三部分构成：

- **本总览**：发布包结构、模型基础配置与两分册导读；
- **第一部分《算法设计文档》**：算法综述（国内外研究现状、SOTA 技术特点、设计思想与先进性）+ 算法设计（架构与数据流、训练数据集构成、模型配置、损失函数设计、基准去泄漏、消融决策记录、参考文献）；
- **第二部分《算法训练文档》**：运行环境、数据处理、训练过程与超参、算法测试（含完整消融实验历程）、计算性能。

消融实验明细数据汇总于附表 `docs/assets/附表A_消融实验汇总.xlsx`（4 个数据表：架构消融 A0–A4 / 输入侧消融 / 结构第三塔 / v3 自评对照）。文中图表按全报告连续编号（图 1–4、表 1–14）。

### 发布包结构与内容分布

```
release.zip
├── final_predictions.xlsx     # v3 模板组内排序回填（predict.py 一键产出，已校验排列合法性）
├── report.docx                # 本文档
└── code/                       # 全部源代码（含 readme 说明）
    ├── requirements.txt        # 固化依赖版本
    ├── train.py                # 一键训练：去泄漏扫描 + 3 种子重训
    ├── inference.py            # 一键推理：逐种子分数 + Borda 集成排序 CSV
    ├── predict.py              # 一键产出最终预测 xlsx（解析 v3 模板 → 推理 → 回填）
    ├── data_pipeline/          # 83 源 CSV 清洗管线（01–07）+ v3 解析 + 08 去泄漏
    ├── extraction/             # 冻结嵌入提取（AntiBERTy 抗体塔 / ESM-2 抗原塔）
    ├── training/               # 模型/训练/评估/推理 + runs/deleak_final_s{0,1,2}（最终检查点）
    └── model_data/             # 预训练权重存放处（开源权重，下载方式见训练文档 §1.2）
```

### 模型基础配置速览

| 项目 | 配置 | 详见 |
|---|---|---|
| 任务形式 | 组内 20 条抗体的亲和力排序（Spearman 评测） | 设计文档 §1 |
| 抗体塔 | AntiBERTy（约 26M，冻结，fp16 缓存） | 设计文档 §2.3 |
| 抗原塔 | ESM-2 650M（冻结，fp16 缓存） | 设计文档 §2.3 |
| 交互 / 融合 | Pre-LN 交叉注意力（Q=抗原）+ 双线性门控 | 设计文档 §2.3 |
| 损失函数 | ListMLE + 并列扰动 / 截断钉底 / rank-only 降权 | 设计文档 §2.4 |
| 可训练参数 | 1,860,737（适配器/交互/融合/打分头） | 设计文档 §2.3 |
| 训练数据 | 641,454 行公开语料，去泄漏后 579,829 行，2,274 合格组 | 设计文档 §2.2/§2.5 |
| 训练成本 | RTX 4090 单卡，约 6 分钟/种子 × 3 种子 | 训练文档 §5 |
| 推理 | CPU 即可，分钟级 | 训练文档 §5 |
| v3 自评 | Spearman 0.667（Group 1: 0.395 / Group 2: 0.938） | 训练文档 §4.5 |

### 两分册内容导读

- **设计文档**回答"为什么这样设计"：§1 综述四条技术路线与 SOTA 特点，给出抗原条件化 + 冻结 PLM 双塔的设计动机（含 PLM 选型的三条理由）；§2 给出架构、数据、损失与去泄漏协议的完整设计，§2.6–2.7 记录输入侧与结构模态两条候选路线的消融否决证据。
- **训练文档**回答"怎么训练与验证"：§1–3 为可复现的运行环境、数据处理与训练配方；§4 以评测协议（3 种子 mean-of-best）开篇，完整记录 A0–A4 架构消融、稳定化研究、输入侧消融、结构第三塔四形态消融与 v3 评测集自评对照；§5 为计算性能与瓶颈分析。

---
"""


def body(path: Path) -> str:
    """Strip the part doc's own H1/header block, keep from the first '## '."""
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^## ", text, flags=re.M)
    if not m:
        raise ValueError(f"no '## ' section found in {path}")
    return text[m.start():].strip()


design = body(DOCS / "算法设计文档.md")
training = body(DOCS / "算法训练文档.md")

out = (
    HEADER
    + "\n# 第一部分 算法设计文档\n\n" + design
    + "\n\n---\n\n# 第二部分 算法训练文档\n\n" + training
    + "\n"
)


# ---------------------------------------------------------------------------
# Cross-reference post-processing (renumbering + anchors + internal links)
# ---------------------------------------------------------------------------

CITE = re.compile(r"(?<!\[)\[(\d{1,2})\](?!\])")
BIB_ENTRY = re.compile(r"^\[(\d{1,2})\]\s+(.*)$")
CAPTION = re.compile(r"^(图|表) (\d{1,2})(?=[\s　:：])")
MENTION = re.compile(r"(?<![\[#])(图|表) (\d{1,2})(?![\d–—-])")
SECREF = re.compile(
    r"(?P<prefix>设计文档|训练文档)?[ ]?"
    r"(?P<ref>§\d+(?:\.\d+)*(?:[–-]\d+(?:\.\d+)*)?)"
)
HEADNUM = re.compile(r"^(#{1,4})\s+(\d+(?:\.\d+)*)\.?(?=\s)")


def add_crossrefs(text: str) -> str:
    lines = text.split("\n")

    # fenced code blocks are never touched
    in_fence = False
    fenced = []
    for ln in lines:
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            fenced.append(True)
            continue
        fenced.append(in_fence)

    p1 = next(i for i, ln in enumerate(lines) if ln.startswith("# 第一部分"))
    p2 = next(i for i, ln in enumerate(lines) if ln.startswith("# 第二部分"))

    def part_of(i: int) -> str:
        return "H" if i < p1 else ("D" if i < p2 else "T")

    # bibliography section: '## 参考文献' .. next heading of level <= 2
    b0 = next(i for i, ln in enumerate(lines) if ln.strip() == "## 参考文献")
    b1 = next(
        i for i in range(b0 + 1, len(lines))
        if re.match(r"^#{1,2} ", lines[i])
    )
    bib_idx = [i for i in range(b0 + 1, b1) if BIB_ENTRY.match(lines[i])]
    entries = {int(m.group(1)): m.group(2) for i in bib_idx
               for m in [BIB_ENTRY.match(lines[i])]}

    # --- 1. renumber by first in-text citation order ----------------------
    order: list[int] = []
    for i, ln in enumerate(lines):
        if fenced[i] or i in bib_idx:
            continue
        for m in CITE.finditer(ln):
            n = int(m.group(1))
            if n not in order:
                order.append(n)
    missing = [n for n in order if n not in entries]
    if missing:
        raise SystemExit(f"citations without bibliography entries: {missing}")
    uncited = [n for n in entries if n not in order]
    if uncited:
        print(f"WARNING: bibliography entries never cited in text: {uncited}"
              " (appended at the end)")
    mapping = {old: new for new, old in enumerate(order + uncited, start=1)}
    print("citation renumbering (old -> new):",
          ", ".join(f"{o}->{n}" for o, n in sorted(mapping.items())))

    for i, ln in enumerate(lines):
        if fenced[i] or i in bib_idx:
            continue
        lines[i] = CITE.sub(lambda m: f"[{mapping[int(m.group(1))]}]", ln)
    renum = {mapping[old]: body_ for old, body_ in entries.items()}
    for slot, new in zip(bib_idx, sorted(renum)):
        lines[slot] = f"[{new}] {renum[new]}"

    # --- 2. anchors on bibliography labels and caption labels -------------
    for i in bib_idx:
        lines[i] = BIB_ENTRY.sub(r"[[\1]]{#ref_\1} \2", lines[i])

    anchors: set[str] = {f"ref_{n}" for n in renum}
    for i, ln in enumerate(lines):
        if fenced[i] or i in bib_idx:
            continue
        m = CAPTION.match(ln)
        if m:
            kind, num = m.group(1), int(m.group(2))
            aid = f"{'fig' if kind == '图' else 'tab'}_{num}"
            anchors.add(aid)
            lines[i] = CAPTION.sub(
                lambda m: f"[{m.group(0)}]{{#{aid}}}", ln, count=1)

    # --- 3. linkify in-text 图 N / 表 N mentions --------------------------
    dangling: list[str] = []
    for i, ln in enumerate(lines):
        if fenced[i] or i in bib_idx:
            continue

        def mention_sub(m: re.Match) -> str:
            kind, num = m.group(1), int(m.group(2))
            aid = f"{'fig' if kind == '图' else 'tab'}_{num}"
            if aid not in anchors:
                dangling.append(f"line {i + 1}: {m.group(0)}")
                return m.group(0)
            return f"[{m.group(0)}](#{aid})"

        lines[i] = MENTION.sub(mention_sub, lines[i])

    # --- 3b. linkify in-text [n] citations --------------------------------
    for i, ln in enumerate(lines):
        if fenced[i] or i in bib_idx:
            continue
        lines[i] = CITE.sub(lambda m: f"[[{m.group(1)}]](#ref_{int(m.group(1))})", ln)

    # --- 4. linkify §x.y section refs + tag referenced headings -----------
    def find_heading(part: str, num: str) -> int | None:
        lo, hi = (0, p1) if part == "H" else ((p1, p2) if part == "D"
                                              else (p2, len(lines)))
        for j in range(lo, hi):
            m = HEADNUM.match(lines[j])
            if m and m.group(2) == num:
                return j
        return None

    unresolved: list[str] = []
    for i, ln in enumerate(lines):
        if fenced[i]:
            continue

        def sec_sub(m: re.Match) -> str:
            prefix = m.group("prefix")
            if prefix:
                part = "D" if prefix == "设计文档" else "T"
            else:
                part = part_of(i)
                if part == "H":  # overview: nearest part name earlier in line
                    before = ln[:m.start()]
                    d, t = before.rfind("设计文档"), before.rfind("训练文档")
                    if max(d, t) >= 0:
                        part = "D" if d > t else "T"
                    else:
                        unresolved.append(f"line {i + 1}: {m.group('ref')}")
                        return m.group(0)
            num = re.match(r"§(\d+(?:\.\d+)*)", m.group("ref")).group(1)
            aid = f"sec{part}_{num.replace('.', '_')}"
            j = find_heading(part, num)
            if j is None:
                unresolved.append(f"line {i + 1}: {m.group(0)}")
                return m.group(0)
            if "{#sec" not in lines[j]:
                lines[j] += f" {{#{aid}}}"
            anchors.add(aid)
            text_ = m.group(0)
            return f"[{text_}](#{aid})"

        lines[i] = SECREF.sub(sec_sub, lines[i])

    # --- 5. one paragraph per bibliography entry --------------------------
    b0 = next(i for i, ln in enumerate(lines) if ln.strip() == "## 参考文献")
    i = b0 + 1
    while i < len(lines):
        if lines[i].startswith("[["):
            if i + 1 < len(lines) and lines[i + 1].startswith("[["):
                lines.insert(i + 1, "")
                i += 2
            else:
                i += 1
        elif lines[i].strip() == "":
            i += 1
        else:
            break

    for msg in dangling:
        print("WARNING dangling fig/tab ref:", msg)
    for msg in unresolved:
        print("WARNING unresolved section ref:", msg)
    print(f"anchors: {len(anchors)}; cross-ref links inserted")
    return "\n".join(lines)


out = add_crossrefs(out)

(DOCS / "report.md").write_text(out, encoding="utf-8")
print("wrote", DOCS / "report.md", f"({len(out)} chars)")
