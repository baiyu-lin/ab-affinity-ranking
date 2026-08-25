# 技术报告 —— 抗原-抗体亲和力排序

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
| 任务形式 | 组内 20 条抗体的亲和力排序（Spearman 评测） | [设计文档 §1](#secD_1) |
| 抗体塔 | AntiBERTy（约 26M，冻结，fp16 缓存） | [设计文档 §2.3](#secD_2_3) |
| 抗原塔 | ESM-2 650M（冻结，fp16 缓存） | [设计文档 §2.3](#secD_2_3) |
| 交互 / 融合 | Pre-LN 交叉注意力（Q=抗原）+ 双线性门控 | [设计文档 §2.3](#secD_2_3) |
| 损失函数 | ListMLE + 并列扰动 / 截断钉底 / rank-only 降权 | [设计文档 §2.4](#secD_2_4) |
| 可训练参数 | 1,860,737（适配器/交互/融合/打分头） | [设计文档 §2.3](#secD_2_3) |
| 训练数据 | 641,454 行公开语料，去泄漏后 579,829 行，2,274 合格组 | [设计文档 §2.2](#secD_2_2)/[§2.5](#secD_2_5) |
| 训练成本 | RTX 4090 单卡，约 6 分钟/种子 × 3 种子 | [训练文档 §5](#secT_5) |
| 推理 | CPU 即可，分钟级 | [训练文档 §5](#secT_5) |
| v3 自评 | Spearman 0.667（Group 1: 0.395 / Group 2: 0.938） | [训练文档 §4.5](#secT_4_5) |

### 两分册内容导读

- **设计文档**回答"为什么这样设计"：[§1](#secD_1) 综述四条技术路线与 SOTA 特点，给出抗原条件化 + 冻结 PLM 双塔的设计动机（含 PLM 选型的三条理由）；[§2](#secD_2) 给出架构、数据、损失与去泄漏协议的完整设计，[§2.6–2.7](#secD_2_6) 记录输入侧与结构模态两条候选路线的消融否决证据。
- **训练文档**回答"怎么训练与验证"：[§1–3](#secT_1) 为可复现的运行环境、数据处理与训练配方；[§4](#secT_4) 以评测协议（3 种子 mean-of-best）开篇，完整记录 A0–A4 架构消融、稳定化研究、输入侧消融、结构第三塔四形态消融与 v3 评测集自评对照；[§5](#secT_5) 为计算性能与瓶颈分析。

---

# 第一部分 算法设计文档

## 1. 算法综述 {#secD_1}

### 1.1 国内外研究现状

抗体-抗原亲和力预测的主流技术路线可分为四类：

1. **零样本蛋白质语言模型（PLM）打分**。ESM-2 [[1]](#ref_1)、IgLM [[2]](#ref_2)、ProGen2 [[3]](#ref_3) 等模型的伪似然/困惑度可直接作为序列适应度代理。但公开基准评测表明，结合亲和力是一种"外在属性"（extrinsic property，依赖于抗原上下文），零样本打分在该任务上很弱：FLAb 基准中所有零样本模型在亲和力数据集上的相关性普遍接近 0 [[4]](#ref_4)；其扩展版 FLAb2 覆盖 30 个模型，最佳模型的平均排序相关也仅 ρ≈0.12，且 PLM 打分存在显著的 germline 偏置（控制 germline 距离后表观预测力平均损失约 40%）[[5]](#ref_5)。零样本打分只能作为基线，不能作为方案。
2. **监督式 PLM 嵌入探针与微调**。AbAffinity 的骨干消融表明，冻结 ESM-2 嵌入 + 轻量预测头是序列骨干中的最强配置，其完整模型在随机划分下达到 Pearson 0.86 / Spearman 0.84 [[6]](#ref_6)；FLAb2 的少样本实验进一步表明，在 embedding 微调设定下**通用蛋白 LM 嵌入不劣于（甚至优于）抗体专用嵌入** [[5]](#ref_5)。这支持"冻结大规模 PLM + 轻量可训练头"的参数高效路线。
3. **排序学习（Learning-to-Rank）**。亲和力数据天然是"同一抗原组内可比、跨组不可比"的偏序数据。列表级排序损失 ListMLE 源于排序学习经典工作 [[7]](#ref_7)；在抗体亲和力任务上，AbLWR 的消融显示将 ListMLE 换为 MSE 回归后 FRA 指标从 21.28% 崩塌至 3.29% [[8]](#ref_8)；AbRank 基准则以成对 margin 排序损失（pairwise margin ranking）构建其基线框架 [[9]](#ref_9)。**排序损失显著优于回归损失**是该领域的共识。
4. **生成式与结构感知路线**。MAGE 用微调语言模型生成抗原特异性配对链抗体并经实验验证 [[10]](#ref_10)；RFdiffusion 面向结构层面的结合体从头设计 [[11]](#ref_11)；IgFold 实现秒级抗体结构预测 [[12]](#ref_12)；SaProt [[13]](#ref_13)、ESM-IF [[14]](#ref_14)、ESM3 [[15]](#ref_15) 等结构感知 PLM 将三维结构信息引入序列建模。这些工作面向"设计/生成"任务或通用表征学习，与本任务"给定抗体集合的亲和力排序"目标不同，列为相邻方向；其中结构感知路线对本任务的适用性，我们以消融实验直接检验（[§2.7](#secD_2_7)）。

双塔冻结编码 + 注意力交互的架构在分子相互作用预测中亦有充分先例：MolTrans（药物-靶点，双通道编码 + cross-view 交互注意力）[[16]](#ref_16)、AttABseq（抗体-抗原突变亲和力变化，注意力机制）[[17]](#ref_17)、DG-Affinity（PLM 嵌入回归亲和力）[[18]](#ref_18)、AbAgIPA（结构感知不变点注意力）[[19]](#ref_19)。

### 1.2 SOTA 算法技术特点

[表 1]{#tab_1} SOTA 算法技术特点与对本任务的适用边界

| 方法 | 类型 | 技术特点 | 对本任务的适用边界 |
|---|---|---|---|
| AbAffinity [[6]](#ref_6) | 序列-only 监督预测 | 冻结 ESM-2 650M 三流嵌入 + CDR 聚焦池化 + 门控交叉注意力，回归 pKd | 证明了冻结 ESM-2 骨干 + 轻量头的有效性；但回归目标与随机划分不直接对应组内排序评测 |
| AbLWR [[8]](#ref_8) | 列表排序学习 | ListMLE + PU 学习 + 同源抗原采样，列表内样本关系建模 | 与本任务目标最吻合（组内排序）；其消融提供了"排序损失不可替代"的直接证据 |
| AbRank / WALLE-Affinity [[9]](#ref_9) | 排序基准 + 基线 | 38 万条结合测定重构为成对排序，GNN + PLM 嵌入 + 结构，OOD 划分 | 提供了严格的去泄漏评估思想；成对 margin 损失是列表损失的替代方案 |
| MAGE [[10]](#ref_10) | 生成式设计 | 微调 PLM 直接生成抗原特异性 VH+VL 配对序列，实验验证 | 面向生成而非排序评测，是后续方向的相邻技术 |
| FLAb/FLAb2 零样本模型群 [[4]](#ref_4)[[5]](#ref_5) | 无训练基线 | 各类 PLM 伪似然直接打分 | 证明零样本路线不可用，同时提供了评测协议与偏置分析范式 |

### 1.3 本算法的设计思想与先进性 {#secD_1_3}

**核心洞察：结合是"外在属性"（extrinsic property）——离开抗原谈抗体亲和力没有意义** [[4]](#ref_4)。仅编码抗体的模型只能学到"每组抗体的质量先验"，无法跨抗原泛化。因此本算法将抗原序列与抗体共同作为一等输入，构建**抗原条件化（antigen-conditioned）**的排序模型。

**为什么选择冻结 PLM 作为双塔（设计初衷）**，三条理由：

1. **表征充分性的公开证据**：大规模 PLM 的序列表征已包含丰富的结构与功能信息——FLAb2 在少样本 embedding 设定下证明通用蛋白 LM 嵌入不劣于抗体专用嵌入 [[5]](#ref_5)；AbAffinity 证明冻结 ESM-2 骨干即可支撑 SOTA 级亲和力预测 [[6]](#ref_6)。无需微调即可获得高质量表征。
2. **数据规模约束**：本任务合格训练组为 2,274 个（每组 ≤20 条），在此数据量上微调 6.5 亿参数的编码器必然过拟合并破坏预训练表征的几何结构；冻结 + 186 万参数轻量头是参数高效且抗过拟合的选择（经消融验证，见[ §2.3](#secD_2_3) 与[训练文档 §4](#secT_4)）。
3. **时间与算力约束**：开发周期内，"嵌入一次性提取落盘 + 仅训练轻量头"使单次训练压缩到分钟级，从而支撑了 3 种子 × 多评估轴 × 十余个消融臂的完整实验迭代（[训练文档 §4](#secT_4)）；微调路线在同等预算下无法完成这种严谨度的验证。

**架构要点**：

- **双塔冻结编码器**：抗体塔 AntiBERTy（约 26M 参数 [[20]](#ref_20)，512 维，在 5.58 亿条 OAS 抗体序列上训练的 MLM 双向编码器）；抗原塔 ESM-2 650M（1280 维 [[1]](#ref_1)，通用蛋白覆盖，可处理任意病毒糖蛋白/多肽抗原）。两塔完全冻结，嵌入一次性提取并落盘缓存，**可训练参数仅 186 万**。
- **抗原作为查询的交叉注意力**：让抗原残基主动"检索"抗体的互补决定区（CDR），符合结合界面的生物学直觉；消融证明该交互模块在多样抗原评测上不可替代（[训练文档 §4.2](#secT_4_2)）。
- **双线性门控融合**：门控单元自适应平衡"被抗体上下文调制后的抗原表征"与"原始抗体表征"（门控思想受 AlphaFold2 Evoformer 门控机制启发 [[21]](#ref_21)），抗原缺失时模型退化为仅抗体路径。
- **ListMLE 列表排序损失**：直接优化评测指标（组内排序）对应的 Plackett-Luce 似然 [[7]](#ref_7)，并针对真实亲和力数据特性做了三项修正（[§2.4](#secD_2_4)）。

**相对开源基线的改进点**：相比零样本 IgLM/ProGen2（ρ≈0，与 FLAb 报告一致 [[4]](#ref_4)），交叉验证 Spearman 提升至 0.30；相比线性探针（pooled 嵌入 + Linear，0.214，且在多样抗原折上崩塌至 0.068），完整交互架构取得 0.276–0.297，**在抗原多样性高的折上优势最大**（+0.21），表明抗原条件化的收益主要体现在泛化评估轴上（明细见[训练文档 §4.2](#secT_4_2)）。

---

## 2. 算法设计 {#secD_2}

### 2.1 算法架构与数据流

![模型架构图](assets/fig_architecture.png)

[图 1]{#fig_1} 模型架构：双塔冻结 PLM（抗体塔 AntiBERTy + 抗原塔 ESM-2 650M）+ 可训练适配器/交叉注意力/双线性门控融合/打分头（共 1,860,737 可训练参数）。

![数据流图](assets/fig_dataflow.png)

[图 2]{#fig_2} 端到端数据流：83 个公开源 CSV → 清洗管线 → 统一语料 → v3 基准去泄漏 → 冻结嵌入缓存 → ListMLE 训练（3 种子）→ Borda 集成 → 产出组内排序；公开真值仅用于独立自评，不进入产出链路（矢量图源文件见 `docs/assets/fig_dataflow.svg`）。

训练：每组采样 ≤20 条列表 → ListMLE 损失 → AdamW [[22]](#ref_22)。推理：逐行（抗体, 抗原）打分 → 按分数降序输出组内 Rank 1–20。

### 2.2 训练数据集构成 {#secD_2_2}

主语料来自公开亲和力数据集的统一清洗管线（`data_pipeline/`，83 个源 CSV → 641,454 行 / 5,697 个抗原组），构成见[表 2](#tab_2)：

[表 2]{#tab_2} 训练数据集构成

| 组成 | 规模 | 说明 |
|---|---|---|
| AbRank 基准集 | 主体 | 中和/逃逸/IC50 等组内排序数据，恒定区已修剪，标签符号审计 |
| kothiwal2025htp | 364 行 / 9 组 | SPR Kd 实测，逐行抗原序列 |
| 其他公开集（SAbDab 等） | 52 组 | 统一协调（harmonize）后并入 |
| 辅助 binder/non-binder 对 | tsuruta / kirby | 消融后未采用（λ_aux=0，[训练文档 §4.2](#secT_4_2)） |

数据清洗关键步骤：恒定区修剪、亲和力标签符号审计、`Aff_op` 截断值（censoring，如 ">100 nM"）解析、仅排序（rank-only）组标记、CDR-H3 聚类与抗原聚类。合格训练组 2,274 个（组内 ≥20 条非截断记录），组权重 min(N, 2000)。

**抗原覆盖**：5,645/5,697 组由管线提供抗原序列，另手工策展 55 条（SAbDab-PDB 优先），共 4,716 条唯一抗原序列；仅 20 组抗原不可解析（走"空抗原"零向量回退路径）。

**防泄漏划分**：留一文件出（LOFO）、AbRank 抗体聚类分层 5 折、抗原聚类留出（整族抗原未见）三重评估轴；跨文件 CDR-H3 重复自动联合留出；随机划分禁用。v3 基准去泄漏见[ §2.5](#secD_2_5)。

### 2.3 模型配置 {#secD_2_3}

[表 3]{#tab_3} 模型配置与选型依据（文献编号见参考文献；消融编号见[训练文档 §4.2](#secT_4_2)）

| 组件 | 配置 | 依据 |
|---|---|---|
| 抗体塔 | AntiBERTy（冻结，缓存）[[20]](#ref_20) | 消融 A1：优于 IgLM [[2]](#ref_2)（0.276 vs 0.261，且更稳定） |
| 抗原塔 | ESM-2 650M（冻结，缓存）[[1]](#ref_1) | 消融 A2 盲抗原对照；FLAb2 少样本证据 [[5]](#ref_5) |
| 适配器 | Linear→LN→GEGLU-FFN+残差，256 维 | GEGLU [[23]](#ref_23)；隐藏层用 softmax/sigmoid 是信息瓶颈（文献共识） |
| 交互 | Pre-LN 交叉注意力，Q=抗原 | 消融 A2：0.276 vs 盲抗原 0.256 vs 仅拼接 0.242；先例 [[16]](#ref_16)[[17]](#ref_17)[[19]](#ref_19) |
| 融合 | 双线性门控 | 消融 A3：≥ 纯拼接；门控机制 [[21]](#ref_21) |
| 损失 | ListMLE [[7]](#ref_7) + 三项修正 | 消融 A4 + 文献 [[8]](#ref_8)[[9]](#ref_9)（[§2.4](#secD_2_4)） |
| 可训练参数 | 1,860,737 | 全部位于适配器/交互/融合/头 |

### 2.4 损失函数设计 {#secD_2_4}

**ListMLE（Plackett-Luce 负对数似然）** [[7]](#ref_7)，针对真实亲和力数据的三项修正：

1. **并列扰动（tie-jitter）**：相同亲和力值在目标序中加微扰，避免任意 tie-breaking 引入的伪序。
2. **截断值钉底（censored pinned）**：">阈值"类截断记录固定排在列表底部，且每个列表截断样本 ≤25%，防止 censored 数据主导梯度。
3. **仅排序组降权（rank_only 0.5×）**：Pred_affinity 等代理标签组按 0.5× 采样降权（消融 A4 证实代理标签会毒化 SPR-Kd 迁移，DCC 折 0.039→0.350，[训练文档 §4.2](#secT_4_2)）。

辅助 VHH margin 损失经消融为中性（0.263 vs 0.261），最终 λ_aux=0，目标函数保持单一 ListMLE。

### 2.5 基准去泄漏 {#secD_2_5}

**事实**：v3 评测集 40 条抗体全部来自公开数据集（FLAb/AbRank 体系），其精确序列与 Kd 可在公开渠道核到（对照表：`flab_reference/benchmark_v3_public_groundtruth.csv`）。原任务说明将 FLAb 数据指定为训练数据来源（AbRank 即其 `data/binding` 组成部分），因此**用 AbRank 训练本身完全合规**；但基准答案天然存在于训练分布中——若不处理，任何模型都可能以"记忆"替代"泛化"，复算时也无法区分两者。

**做法**：用同一配方训练"剔除 / 未剔除"两臂，直接观察模型在未见答案时的真实泛化能力（对照结果见[训练文档 §4.5](#secT_4_5)）。

**去泄漏方法**（`data_pipeline/08_deleak_v3.py`，全自动可追溯）：

1. 对 40 条基准抗体做 (VH, VL) 精确匹配，命中 **40/40**（共 64 个语料行）；
2. 按 CDR-H3 聚类（`ab_cluster_raw`）做**同簇扩展剔除**，无簇归属的命中行按行剔除：合计剔除 39 簇 + 16 行孤儿 = **101 行**（全部带亲和力标签；来源：AbRank 85 行、rawat2022abcov_kd 7 行、shanehsazzadeh2023 三文件 8 行），剔除清单落盘 `deleak_v3_hits.csv`；
3. 剔除在训练加载时生效（`exclude_json` 配置项），语料文件本体不动，可随时审计/回滚；未剔除对照臂仅需去掉该配置项复训（同环境已复现，见[训练文档 §4.5](#secT_4_5)）；
4. 嵌入缓存只含序列→向量映射，删行不影响缓存有效性，直接复用，仅重训 186 万参数头部。

### 2.6 输入侧架构决策记录 {#secD_2_6}

v3 重建时对输入接口做了两个候选方向的系统验证，结论：**维持扁平拼接接口 + AntiBERTy/ESM-2 双塔**（详细数据见[训练文档 §4.3](#secT_4_3) 与附表 xlsx）。

**候选一：显式链感知接口（chain_ids）**。背景：数据层全程保留 VH/VL 分列与链长，但模型接口把两条链拼接为一条扁平序列，无显式链边界。消融臂实现：collate 生成 `chain_ids [B,L]`（0=VH，1=VL），叠加 chain-type embedding 到适配器输入。结果：

- 监控集 Spearman 逐种子对比：0.787/0.749/0.666 vs 基线 0.804/0.766/0.700，**3/3 种子全劣**；
- v3 公开真值自评（集成）：**0.532** vs 基线 **0.714**（Group 1：0.278 vs 0.585；Group 2：0.786 vs 0.842）。

判读：对冻结 PLM 表征强行叠加可学习的链型偏置，破坏了预训练表征的既有几何，新参数在小数据上引入噪声。**链感知臂否决。**

**候选二：混合编码器塔（条件触发）**。触发判据：仅 VH / 仅 VL 输入的得分退化对比（`training/probe_chains.py`，对最终模型做掩码探针），结果见[表 4](#tab_4)：

[表 4]{#tab_4} 链掩码探针：仅 VH / 仅 VL 输入的得分退化（与公开真值的 Spearman）

| 输入变体 | Group 1（双链均变异） | Group 2（VL 全同） |
|---|---|---|
| 完整输入 | 0.593 | 0.845 |
| 仅 VH | 0.382 | 0.833 |
| 仅 VL | 0.075 | ≈0（常数） |

判读：VH 承载主导排序信号（Group 2 VL 无组内变异，仅 VL 自然归零，符合设计）；Group 1 上 VL 掩码损失 −0.52、VH 掩码损失 −0.21，**双链互补且 VL 信息已被模型有效利用**——分析未指向 AntiBERTy 对某条链的表征不足，混合塔（AntiBERTy+ESM-2 混编等）备选臂**不启动**。

### 2.7 结构模态第三塔消融 {#secD_2_7}

**动机**：结构感知预训练模型（SaProt [[13]](#ref_13)、ESM-IF [[14]](#ref_14)、ESM3 [[15]](#ref_15) 等）在若干功能预测任务上报告了增益，且结构信息与序列信息在表征层面上属于不同视角。本节检验：在双塔之外注入结构嵌入（第三塔）能否为本排序任务带来增量。

**共同协议**：与最终模型同语料（v3 去泄漏后）、同配方（10 轮 × 1500 步、3 种子、监控集 100 组取 best Spearman），仅新增结构支路；结构特征全部离线提取并落盘缓存，可训练部分仍仅为轻量连接层，与基线的唯一差异即结构输入。

**真实结构可行性预检（Arm 0）**：配套的 SAbDab 结构数据中，抗原侧 25 条唯一序列 23 条精确命中，但**抗体侧真实结构覆盖率仅 0.22%**（314/144,140 唯一 VH 精确匹配）——"收集现有真实结构"路线在抗体侧不可行，以下各臂结构特征均来自预测结构（IgFold [[12]](#ref_12) 折叠 → foldseek 3Di 结构词表）。

四种融合形态：池化向量经 Linear 残差注入塔摘要向量（残差臂）、池化向量重塑为 8 个 struct token 后由塔摘要作 Query 交叉注意力（交叉注意力臂）、逐残基特征在抗体塔 FFN 之前以零初始化门控残差注入（逐残基门控臂，ReZero 式 [[24]](#ref_24)，训练起点严格等于基线）。消融结果见[表 5](#tab_5)（注入位置与实现明细见附表 xlsx sheet 3）：

[表 5]{#tab_5} 结构模态第三塔消融结果（监控集 best Spearman，3 种子）

| 臂 | 结构编码器 | 融合粒度 | s0 | s1 | s2 | 均值 | Δ vs 基线 |
|---|---|---|---|---|---|---|---|
| **基线** | — | — | 0.8042 | 0.7663 | 0.7003 | **0.7569** | — |
| ProstT5 残差 | ProstT5 [[25]](#ref_25) | 池化 | 0.7983 | 0.7595 | 0.7005 | 0.7528 | −0.004 |
| SaProt 残差 | IgFold + SaProt-650M [[13]](#ref_13)[[12]](#ref_12) | 池化 | 0.8042 | 0.7661 | 0.7025 | 0.7576 | +0.001 |
| SaProt 交叉注意力 | 同上 | 池化 | 0.7947 | 0.7387 | 0.6972 | 0.7435 | −0.013 |
| SaProt 逐残基门控 | 同上 | 逐残基 | 0.8012 | 0.7710 | 0.6957 | 0.7560 | −0.001 |

补充 OOD 证据（ProstT5 臂，AbRank fold0 抗原聚类留出）：0.2727 vs 基线 0.2513，+0.021 未达预设 +0.03 采纳门槛，且在种子噪声（基线逐种子散布 0.046）之内。

> 注：本表基线为与各消融臂同批次的 deleak_final 快照；最终检查点为同配方复训为同配方复训（[训练文档 §3.3](#secT_3_3)：0.804/0.767/0.707），与快照的差异属 GPU 非确定性噪声，不影响消融判定（各臂 Δ 均 ≤ 噪声量级或为显著负值）。

**判读**：

- 四种形态覆盖了**两条独立结构编码路径**（免结构预测的 ProstT5 [[25]](#ref_25) vs 显式 IgFold 结构 + SaProt [[13]](#ref_13)[[12]](#ref_12)）与**三种注入方式**（池化残差 / 池化交叉注意力 / 逐残基门控塔内融合），结果全部落在种子噪声内或显著为负；
- 逐残基门控臂是对"注入点/粒度"假设的最强检验：结构信息在抗体 token 与抗原交叉注意力交互**之前**逐残基融入，且零初始化门控 [[24]](#ref_24) 保证不扰动基线起点——仍为零增益，说明瓶颈不在融合方式；
- 训练动态佐证：残差臂逐 epoch 曲线与基线几乎重合，模型学到的近乎恒等映射，结构支路贡献趋零。

**负结果的解读（信息视角的假设）**：回顾候选结构编码路径可以发现，**它们的输入在信息论意义上仍以序列为唯一来源**——IgFold 的三维结构本身就是从序列预测得来的派生产物，3Di 结构词表与 SaProt 的"结构感知"嵌入是对这一派生产物的再编码，整条链路上并没有引入任何序列之外的新信息。而本任务的评测目标是**同一抗原/同一抗体框架内细微亲和力差异的组内排序**：这类信号主要由 CDR  loop 的氨基酸组成与相互作用决定，序列嵌入（AntiBERTy/ESM-2）在此已接近饱和；结构通路所能提供的仅是折叠家族层级的粗粒度先验——对高度同源的组内变体而言近乎常量，构成**冗余甚至无信息的表征**。零初始化门控臂中模型学到近乎恒等映射，正是这一解读的机制层面印证。换言之，负结果的原因不是"结构信息无用"，而是**在当前链路下结构模态不包含序列之外的增量信息**；真正的实验测定结构（或端到端结构微调）是否不同，留待后续工作验证。

---

## 参考文献

[[1]]{#ref_1} Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 2023, 379(6637): 1123-1130.

[[2]]{#ref_2} Shuai R W, Ruffolo J A, Gray J J. IgLM: Infilling language modeling for antibody sequence design. Cell Systems, 2023, 14(11): 979-989.

[[3]]{#ref_3} Nijkamp E, Ruffolo J A, Weinstein E N, et al. ProGen2: Exploring the boundaries of protein language models. Cell Systems, 2023, 14(11): 968-978.

[[4]]{#ref_4} Chungyoun M, Ruffolo J A, Gray J J. FLAb: Benchmarking deep learning methods for antibody fitness prediction. bioRxiv, 2024. DOI: 10.1101/2024.01.13.575504.

[[5]]{#ref_5} Chungyoun M, Gray J J. Fitness Landscape for Antibodies 2: Benchmarking reveals that protein AI models cannot yet consistently predict developability properties. bioRxiv, 2025. PMID: 41497662.

[[6]]{#ref_6} Singh H, Malhotra A, Srivastava S P, et al. Antibody-antigen affinity prediction with chain-aware protein language modeling (AbAffinity). bioRxiv, 2026. DOI: 10.64898/2026.06.19.733375.

[[7]]{#ref_7} Xia F, Liu T Y, Wang J, et al. Listwise approach to learning to rank: theory and algorithm. Proceedings of ICML, 2008: 1192-1199.

[[8]]{#ref_8} Xu F, Huang Z A, He H, et al. AbLWR: A context-aware listwise ranking framework for antibody-antigen binding affinity prediction via positive-unlabeled learning. arXiv:2604.11272, 2026.

[[9]]{#ref_9} Liu C, Pelissier A, Shao Y, et al. AbRank: A benchmark dataset and metric-learning framework for antibody-antigen affinity ranking. arXiv:2506.17857, 2025.

[[10]]{#ref_10} Wasdin P T, Johnson N V, Janke A K, et al. Generation of antigen-specific paired-chain antibodies using large language models (MAGE). Cell, 2025, 188(25): 7206-7221.

[[11]]{#ref_11} Watson J L, Juergens D, Bennett N R, et al. De novo design of protein structure and function with RFdiffusion. Nature, 2023, 620(7976): 1089-1100.

[[12]]{#ref_12} Ruffolo J A, Chu L S, Mahajan S P, et al. Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies (IgFold). Nature Communications, 2023, 14: 2389.

[[13]]{#ref_13} Su J, Han C, Zhou Y, et al. SaProt: Protein language modeling with structure-aware vocabulary. ICLR, 2024.

[[14]]{#ref_14} Hsu C, Verkuil R, Liu J, et al. Learning inverse folding from millions of predicted structures (ESM-IF). Proceedings of ICML (PMLR 162), 2022: 8946-8970.

[[15]]{#ref_15} Hayes T, Rao R, Akin H, et al. Simulating 500 million years of evolution with a language model (ESM3). Science, 2025, 387(6736): 850-858.

[[16]]{#ref_16} Huang K, Xiao C, Glass L M, Sun J. MolTrans: Molecular interaction transformer for drug-target interaction prediction. Bioinformatics, 2021, 37(6): 830-836.

[[17]]{#ref_17} Jin R, Ye Q, Wang J, et al. AttABseq: An attention-based deep learning prediction method for antigen-antibody binding affinity changes based on protein sequences. Briefings in Bioinformatics, 2024, 25(4): bbae304.

[[18]]{#ref_18} Yuan Y, Chen Q, Mao J, et al. DG-Affinity: Predicting antigen-antibody affinity with language models from sequences. BMC Bioinformatics, 2023, 24: 430.

[[19]]{#ref_19} Gu M, Yang W, Liu M. Prediction of antibody-antigen interaction based on backbone aware with invariant point attention (AbAgIPA). BMC Bioinformatics, 2024, 25: 348.

[[20]]{#ref_20} Ruffolo J A, Gray J J, Sulam J. Deciphering antibody affinity maturation with language models and weakly supervised learning. arXiv:2112.07782, 2021（NeurIPS 2021 MLSB Workshop）.

[[21]]{#ref_21} Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. Nature, 2021, 596(7873): 583-589.

[[22]]{#ref_22} Loshchilov I, Hutter F. Decoupled weight decay regularization. ICLR, 2019.

[[23]]{#ref_23} Shazeer N. GLU variants improve transformer. arXiv:2002.05202, 2020.

[[24]]{#ref_24} Bachlechner T, Majumder B P, Mao H, et al. ReZero is all you need: Fast convergence at large depth. Proceedings of UAI (PMLR 161), 2021.

[[25]]{#ref_25} Heinzinger M, Weissenow K, Sanchez J G, et al. Bilingual language model for protein sequence and structure (ProstT5). NAR Genomics and Bioinformatics, 2024, 6(4): lqae150.

---

# 第二部分 算法训练文档

## 1. 运行环境 {#secT_1}

### 1.1 硬件

[表 6]{#tab_6} 硬件环境

| 用途 | 配置 |
|---|---|
| 嵌入提取（一次性）+ 推理 | CPU 即可（无 GPU 依赖）；GPU 可加速 |
| 模型训练 | 云端 GPU 实例：RTX 4090 24GB ×1，16 vCPU（Xeon Platinum 8352V），120GB 内存，Ubuntu 22.04 |

### 1.2 软件依赖

[表 7]{#tab_7} 核心软件依赖与版本

| 库 | 训练端（GPU） | 推理/提取端（CPU） |
|---|---|---|
| Python | 3.12 | 3.12（venv） |
| PyTorch | 2.8.0+cu128（CUDA 12.8） | 2.13.0+cpu |
| transformers | 5.14.1 | 5.14.1 |
| numpy / pandas / scipy | — | 2.5.1 / 3.0.3 / 1.18.0 |
| tokenizers / openpyxl | — | 0.22.2 / 3.1.5 |

安装：`pip install -r code/requirements.txt`（版本已固化）。预训练权重（开源，下载至 `code/model_data/`）：AntiBERTy（HuggingFace `zrollins/antiberty` 镜像，含 `vocab.txt`）、ESM-2 650M（`facebook/esm2_t33_650M_UR50D`）。

### 1.3 运行命令示例

```bash
# 一键训练：基准去泄漏扫描 -> 3 种子重训（含运行结果案例注释）
python code/train.py

# 一键推理：v3 评测集 -> 逐种子分数/排名 + Borda 集成 CSV
python code/inference.py
# 输出 code/predictions/mAbs_ranking.csv（40 行，2 组各为 1–20 排列）

# 一键产出最终文件：解析 v3 xlsx -> 集成推理 -> 产出组内排序
python code/predict.py
# 输出 final_predictions.xlsx（保留 v3 模板结构与格式）

# 单元冒烟测试（合成缓存 + 各消融臂 20 步）
python code/training/smoke_test.py
```

完整管线（如需从零重建语料/缓存）：`data_pipeline/01–07` → `extraction/extract_antiberty.py` + `extraction/extract_antigens.py` → `code/train.py`。

## 2. 数据处理

### 2.1 训练数据输入/输出格式

- **输入**：83 个公开数据集源 CSV（AbRank、kothiwal2025htp、SAbDab 衍生集等），字段各异。
- **管线输出**（`data_pipeline/output/`）：
  - `harmonized_clustered.csv.gz` — 641,454 行 / 5,697 抗原组；统一字段（`seq_heavy/seq_light/fitness/censored/rank_only/ab_cluster_raw/ag_cluster_raw/ag_seq` 等）；
  - `training_groups.csv` — 2,274 个合格训练组（权重 min(N,2000)）；
  - `splits.json` — LOFO 折、AbRank 聚类分层 5 折、抗原聚类留出折、跨文件泄漏标记；
  - `deleak_v3_exclude.json` — v3 基准去泄漏剔除清单（[设计文档 §2.5](#secD_2_5)）。
- **嵌入缓存**（fp16 npz，按 sha1 去重）：
  - `extraction/cache_antiberty/` — 235,331 条唯一抗体（VH+VL 逐链过 AntiBERTy，去特殊符后重→轻拼接，[L,512]，npz 内含 `len_heavy` 链边界）；
  - `extraction/cache_esm2/` — 4,716 条唯一抗原（[L,1280]；>1000 aa 滑窗 1000/500 重叠平均）；缺失抗原走零向量回退。

### 2.2 v3 评测集处理

v3 评测集 xlsx（mAbs 表）→ `data_pipeline/parse_v3_xlsx.py` 一键解析为 `data/benchmark_mAbs_v3.csv`（40 行：group, seq_id, antigen_name, antigen, VH, VL），解析内置断言：40 行、每组 20 条、(group, seq_id) 唯一、Group 2 VL 全同、全大写氨基酸字母表。v3 CSV 只由该脚本生成，保证可追溯。

- **Group 1**：SARS-CoV-2 全长刺突蛋白（1273 aa，全组共享，走滑窗路径）× 20 条 mAb；
- **Group 2**：HER2 胞外域（630 aa）× 20 条曲妥珠框架 CDR-H3 变体（VL 全部相同）。

推理时**每行抗体与其自身抗原配对打分**，与训练格式一致；组大小自适应（不写死 20/组），输出断言每组为 1–N 排列。

### 2.3 自有数据集说明

未使用私有数据；全部训练数据来自公开基准（[设计文档 §2.2](#secD_2_2)）。数据侧的贡献在于统一清洗（恒定区修剪、标签符号审计、截断值解析、聚类防泄漏）与 v3 基准去泄漏协议（[设计文档 §2.5](#secD_2_5)）。

## 3. 算法训练

### 3.1 训练过程

冻结双塔 → 嵌入缓存 → 仅训练适配器/交互/融合/头（1,860,737 参数）。每个训练步：按组权重采样一个组 → 组内取 ≤20 条列表（截断样本 ≤25%）→ 加载缓存嵌入 → 前向打分 → ListMLE 损失（并列扰动 + 截断钉底）→ AdamW 更新 [[22]](#ref_22)。

**最终模型（v3 去泄漏版）**：在去泄漏后的 579,829 条合格行上训练（= 579,930 − 101 剔除行；泄漏守卫另自动剔除跨文件重复 61,524 行），3 个随机种子，每种子保留 `best.pt`（监控集最优轮）与 `last.pt`。

### 3.2 超参数

[表 8]{#tab_8} 超参数与确定依据

| 超参数 | 取值 | 确定依据 |
|---|---|---|
| 优化器 | AdamW [[22]](#ref_22)，β=(0.9,0.999)，wd 0.01（仅权重矩阵） | 计划基线 |
| 学习率 | 3e-4；融合/门控参数组 1e-3（无 wd） | 稳定化研究：1e-4 欠训（0.250），3e-4 保留（[§4.2](#secT_4_2)） |
| 步数/轮 | 1500（由 500 上调，+0.021 均值，4/5 折获胜） | 稳定化研究（3 种子 × 5 折，[§4.2](#secT_4_2)） |
| 轮数 / 早停 | 10 轮，耐心 3（按验证 Spearman，非损失） | — |
| 调度 | 余弦退火 + 5% warmup，梯度裁剪 1.0 | — |
| 列表大小 / 截断上限 | 20 / 25% | 评测格式 |
| rank_only 降权 | 0.5× | 消融 A4（代理标签毒化证据，[§4.2](#secT_4_2)） |
| λ_aux | 0.0（辅助 margin 损失关闭） | 消融 A4b：中性（[§4.2](#secT_4_2)） |
| 批次 | token 预算 65,536 | 显存/内存约束 |

### 3.3 收敛性分析 {#secT_3_3}

去泄漏重训 3 个种子（10 轮 × 1500 步）均完整跑满，收敛情况见[表 9](#tab_9)、[图 3](#fig_3)：

[表 9]{#tab_9} 去泄漏重训收敛情况（3 种子）

| 种子 | 末轮训练损失 | 监控集 Spearman（最优轮） | 最优轮 |
|---|---|---|---|
| 0 | 1.824 | 0.804 | 9 |
| 1 | 1.822 | 0.767 | 9 |
| 2 | 1.824 | 0.707 | 9 |

![损失收敛曲线](assets/fig_loss_curves.png)

[图 3]{#fig_3} 训练损失与监控集 Spearman 随轮次曲线（去泄漏重训，3 种子）。损失平滑下降无发散；监控 Spearman 总体爬升、个别轮次小幅回落（监控集在训练集内，数值偏乐观，仅用于选轮）。种子间差异主要来自初始化轨迹噪声——这正是采用多种子集成的原因（[§4.1](#secT_4_1)）。

## 4. 算法测试与消融实验 {#secT_4}

### 4.1 评测协议 {#secT_4_1}

单折单种子的验证 Spearman 轮间摆动达 ±0.1，同一折同一种子在 CPU 与 GPU 初始化下相差 ~0.04——单次运行的数字不用于架构决策。因此确立两条协议：

1. **所有消融判定采用 3 种子 × 每种子取最优轮（mean-of-best）**，且对比臂与基线同批次、同配方训练，唯一差异即被消融的组件；
2. **最终推理采用 3 种子 Borda 集成**（组内逐种子排名取平均再排序），进一步抑制轨迹噪声。

另有两条工程约束：嵌入缓存经完整性扫描后方可用于训练（同一缓存同一时刻仅一个写入进程）；抗原聚类留出折 1–4 因验证组数偏斜（47–81 组 vs fold0 的 1,989 组）不具判别力，OOD 判定仅使用 fold0。

### 4.2 架构消融历程 A0–A4 与稳定化研究 {#secT_4_2}

以下为模型定型（v5 配方）的完整消融矩阵，协议：AbRank 聚类分层 5 折 + 抗原聚类留出 fold0，3 种子 mean-of-best（明细见附表 sheet 1）。

**A0 — 完整模型 vs 基线**（[表 10](#tab_10)）：

[表 10]{#tab_10} A0 消融：完整模型 vs 基线（AbRank 5 折均值）

| 基线 | AbRank 5 折均值 | 备注 |
|---|---|---|
| 线性探针（pooled 512+1280 → Linear） | 0.214 | fold0 崩塌至 0.068——无跨抗原迁移 |
| 零样本 IgLM / ProGen2 | ≈0 / 仅 kothiwal-DCC 0.322 | 地板线，与 FLAb 报告一致 [[4]](#ref_4) |
| **完整模型（交叉注意力+双线性门控）** | **0.276** | 较探针 +0.062，5 折中胜 4 折 |

**A1 — 抗体塔选型**：AntiBERTy 0.276 ± 0.064（fold0 0.083） vs IgLM 0.261 ± 0.059（fold0 0.058）。IgLM 的因果 LM 特征最不稳定（fold0 种子散布 0.094–0.257，三次 epoch-0 早停）。**AntiBERTy 入选** [[20]](#ref_20)[[2]](#ref_2)。

**A2 — 交互方式**（[表 11](#tab_11)）：

[表 11]{#tab_11} A2 消融：交互方式对比（3 种子 mean-of-best）

| 臂 | AbRank 均值 | 抗原留出 fold0 |
|---|---|---|
| **交叉注意力（Q=抗原）** | **0.276 ± 0.064** | **0.083 ± 0.051** |
| 盲抗原对照（去掉抗原塔） | 0.256 ± 0.075 | 0.054 ± 0.026 |
| 仅拼接（无交互） | 0.242 ± 0.089 | 0.075 ± 0.015 |

交叉-盲抗原差距在 AbRank 上温和（+0.020），在最严格留出轴上接近翻倍（0.083 vs 0.054）；仅拼接在 0/1/3 折落败，交互模块保留。

**A3 — 融合方式**：双线性门控 ≥ 纯拼接（两个评估轴均值均占优），保留。

**A4 — 数据因子**：

- **rank_only 代理标签**：完全排除（factor=0）对 SPR-Kd 迁移是决定性的（DCC 折 0.039 → 0.350），但对 AbRank 式训练显著有害（−0.011 vs 0.276——AbRank 合格组大多是 rank-only）。**保留 0.5× 降权**作为折中；
- **λ_aux（VHH margin 辅助损失）**：0.263 vs 0.261，中性。**默认改为 0.0**，目标函数保持单一 ListMLE。

**稳定化研究（配方定型）**：针对轮间 ±0.1 摆动与 epoch-0 早停现象，AbRank 5 折 × 种子 {0,1}，结果见[表 12](#tab_12)：

[表 12]{#tab_12} 稳定化研究：训练配方对比

| 配方 | 均值 ± sd |
|---|---|
| 基线（lr 3e-4，500 步/轮） | 0.276 ± 0.064 |
| lr 1e-4（融合 3e-4） | 0.250 ± 0.088 ——欠训，否决 |
| **1500 步/轮** | **0.297 ± 0.066 ——采纳**（4/5 折获胜，达预设 ±0.02 门槛） |

机制解读：更多更新步数降低了"首轮评估碰巧走好"的运气依赖。**`steps_per_epoch` 500 → 1500。**

**难边界**：所有臂在整族抗原未见（抗原聚类留出 fold0）时 Spearman 均 ≪0.1（最优 0.083）vs AbRank 聚类折的 0.28——模型学到的是抗原*特异*的结合上下文而非可迁移的抗原生物学。这决定了对未见抗原评测组的现实预期（[§4.6](#secT_4_6) 不足分析）。

### 4.3 输入侧消融 {#secT_4_3}

明细见[设计文档 §2.6](#secD_2_6) 与附表 sheet 2，结论摘要：

- **链感知接口（chain_ids）**：监控集 3/3 种子全劣（0.787/0.749/0.666 vs 0.804/0.766/0.700），v3 自评 0.532 vs 0.714。**否决**——可学习链型偏置破坏了冻结表征的既有几何；
- **链掩码探针**：仅 VH 输入保留大部分信号（Group 1 0.382 vs 完整 0.593；Group 2 0.833 vs 0.845），仅 VL 接近归零（Group 2 VL 无组内变异，符合设计）。双链互补、VL 信息已被有效利用，**混合编码器塔备选臂不启动**。

### 4.4 结构模态第三塔消融 {#secT_4_4}

明细见[设计文档 §2.7](#secD_2_7) 与附表 sheet 3。摘要：ProstT5 池化残差 −0.004、SaProt 池化残差 +0.001、SaProt 池化交叉注意力 −0.013、SaProt 逐残基门控塔内融合 −0.001（均 vs 同协议基线 0.7569，3 种子 mean-of-best）；OOD 补充（ProstT5 臂，fold0）+0.021 未达采纳门槛。**结构第三塔路线整体证伪，不进入最终管线**；负结果的信息视角解读见[设计文档 §2.7](#secD_2_7)。

工程注记：SaProt 逐残基缓存为 235,331 条抗体全量提取（IgFold 折叠 → foldseek 3Di → SaProt-650M 前向，约 24 小时，0 失败）；真实结构预检（Arm 0）显示抗体侧 SAbDab 精确覆盖率仅 0.22%，证实"收集现有真实结构"不可行。

### 4.5 v3 评测集结果 {#secT_4_5}

最终结果文件由 `predict.py` 产出：解析 v3 xlsx → 3 种子去泄漏检查点 Borda 集成 → 产出组内排序（每组 1–20 排列，已校验）。利用基准的公开真值（Kd）做**独立自评**（仅校验）：

![v3 自评对比](assets/fig_benchmark.png)

[图 4]{#fig_4} v3 基准独立自评：去泄漏重训（最终模型）vs 未去泄漏对照臂 vs 链感知消融臂（3 种子 Borda 集成，与公开 −log Kd 的组内 Spearman）。

[表 13]{#tab_13} v3 评测集自评结果（与公开真值的组内 Spearman）

| 模型 | Group 1（SARS-CoV-2） | Group 2（HER2） | 均值 |
|---|---|---|---|
| **去泄漏重训（最终模型，3 种子集成）** | **0.395** | 0.938 | **0.667** |
| 未去泄漏对照臂（同环境复训，仅对照） | 0.050 | 0.931 | 0.490 |
| 链感知消融臂（否决，见[ §4.3](#secT_4_3)） | 0.278 | 0.786 | 0.532 |

逐种子明细（去泄漏模型，分数与公开 −log Kd 的 Spearman）：Group 1 = 0.287 / 0.304 / 0.155，Group 2 = 0.750 / 0.874 / 0.910——Group 2 三种子一致且集成后升到 0.938，Group 1 种子间离散较大（明细见附表 sheet 4）。

对照解读（去泄漏 vs 未去泄漏，同语料/同缓存/同配方，唯一差异为 101 行剔除）：未去泄漏臂在 Group 2 上 0.931 ≈ 去泄漏臂 0.938——该组本就有同框架真实 Kd 训练信号，剔除基准行几乎不损失泛化能力；Group 1 上未去泄漏臂反而降至 0.050（逐种子 0.072 / −0.069 / 0.000）——见过基准精确标签并未转化为组内排序能力。两臂均值 0.667 vs 0.490。（历史对照：首轮旧实例上两臂分别为 0.585/0.842/0.714 与 0.457/0.953/0.705；Group 1 跨环境波动达 0.05–0.59，该组评估噪声较大，见[ §4.6](#secT_4_6)。）

### 4.6 优势与不足 {#secT_4_6}

- **优势**：抗原条件化带来的跨抗原泛化（多样抗原折 +0.2 以上）；可训练参数量小（186 万），训练快、抗过拟合；针对截断/代理标签数据的损失修正；抗原缺失时退化为仅抗体路径；基准解析、去泄漏、训练、推理、回填各环节均脚本化、可复现审计；结构模态经四形态消融确认无增量后未纳入最终模型（[§4.4](#secT_4_4)）。
- **不足**：整族抗原未见时（抗原聚类留出）Spearman 仅 ~0.08，模型学到的是抗原特异的结合上下文而非可迁移的抗原生物学；Group 1（全长刺突、中和表位多样）排序置信度中等，种子间方差大；逐行抗原依赖提供抗原序列，缺失时退化为抗体先验。

## 5. 计算性能 {#secT_5}

[表 14]{#tab_14} 各阶段资源消耗与耗时

| 阶段 | 资源 | 耗时/开销 |
|---|---|---|
| 抗体嵌入提取（235,331 条，一次性） | RTX 4090（GPU 版脚本，~640 条/s） | 48 GB fp16 缓存，**约 6 分钟** |
| 抗原嵌入提取（4,716 条，一次性） | 同上（~51 条/s） | 4.9 GB fp16 缓存，平均 ~430 aa/条，11 条滑窗，**约 2 分钟** |
| 训练（单种子 10 轮 × 1500 步，RTX 4090） | 显存 ~3.5 GB，RAM ~3.3 GB | 25–51 s/轮，**全程 4.5–8.5 分钟** |
| 去泄漏重训（3 种子合计） | 同上 | **约 18 分钟**（缓存复用，仅训头部） |
| 推理（40 抗体 + 2 抗原 + 3 检查点） | CPU <4 GB RAM / GPU | 分钟级（嵌入提取 + 打分 + 回填） |

（附）结构消融臂的额外一次性成本（不进最终管线）：SaProt 逐残基提取约 24 CPU/GPU 小时（127 GB 缓存）、池化版约 27 小时；每消融臂训练约 35 GPU 分钟（3 种子）。

瓶颈算子：训练期为嵌入 npz 的随机读取与 padding 批次的前向（256 维小矩阵乘，主要受 CPU 数据供给限制，GPU 利用率 ~47%）；推理期为 ESM-2 650M 的前向（650M 参数，CPU 上单条约秒级，全长刺突滑窗 ×3）。因双塔冻结且结果缓存，端到端成本集中在一次性提取阶段；训练/推理本身均为分钟级。

---

## 附：复现说明

发布包结构见报告总览（"发布包结构与内容分布"）。

预训练双塔为开源模型（AntiBERTy / ESM-2 650M），全程冻结未微调；自训练部分为适配器/交互/融合/打分头（1,860,737 参数）。选择冻结的理由与证据见[设计文档 §1.3](#secD_1_3)；头部自训练满足"自训练模型"认定（最终模型版本收录于 `code/training/runs/`）。
