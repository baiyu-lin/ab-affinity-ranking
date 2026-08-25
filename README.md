# 抗原-抗体亲和力排序 —— 抗原条件化双塔冻结 PLM + 交叉注意力 + ListMLE

**算法**：双塔冻结蛋白质语言模型（AntiBERTy 抗体塔 + ESM-2 650M 抗原塔）+ 交叉注意力交互 + 双线性门控融合 + ListMLE 列表排序学习。可训练参数 186 万。最终方案见 `archive_2026-08-08/docs/plan-v4.md`；代码文档见 `code/` 内各 README（`code/data_pipeline/README.md`、`code/extraction/README.md`、`code/training/README.md`、`code/training/runs-README.md`）。

## 目录结构

```
.
├── code/                              # 完整代码运行环境（数据管线→嵌入提取→训练→推理）
│   ├── data_pipeline/                 # 83 源 CSV → 641,454 行统一语料 + 防泄漏划分
│   ├── extraction/                    # 冻结编码器嵌入提取（AntiBERTy / ESM-2）
│   ├── tools/                         # 注册式模块库（注意力/激活单元与块参数）
│   ├── training/                      # 模型、训练、评估、推理代码 + 配置
│   ├── data/benchmark_mAbs_v3.csv     # 标准测试集（mAbs 表解析结果）
│   ├── predictions/                   # 推理输出（mAbs_ranking.csv）
│   └── model_data/                    # 预训练权重存放处（需自行下载，见下）
├── models/                            # 最终模型（3 种子 best.pt + 训练日志）
│   ├── deleak_final_s{0,1,2}/
│   └── noleak_final_s{0,1,2}/
├── predictions/                       # 最终 v3 结果（排序 CSV 与自评 JSON）
└── archive_2026-08-08/                # 归档：最终方案 plan-v4.md、基准数据、旧模型与训练记录
```

注：v3 基准 xlsx 不入库（需另行获取）；最终预测文件由 `code/predict.py` 一键产出。

## 依赖与环境

- Python 3.10+；训练端：PyTorch 2.8.0+cu128（CUDA 12.8，RTX 4090）；推理/提取端：PyTorch 2.13.0+cpu
- `pip install torch transformers==5.* pandas numpy scipy tokenizers openpyxl`
- 预训练权重（开源，需下载到 `code/model_data/`）：
  - AntiBERTy：`model_data/antiberty/`（HuggingFace `zrollins/antiberty` 镜像，含 `vocab.txt`）
  - ESM-2 650M：`model_data/esm2-650m/`（HuggingFace `facebook/esm2_t33_650M_UR50D`）

## 运行命令

```bash
cd code
python -m venv .venv && .venv/bin/pip install torch transformers pandas numpy scipy tokenizers openpyxl

# 演示案例（端到端）：对 mAbs 测试集提取嵌入并用 3 种子集成推理
# （需先将 models/deleak_final_s{0,1,2}/best.pt 复制到 code/training/runs/ 下同名目录）
.venv/bin/python training/infer_benchmark.py
# 输出 code/predictions/mAbs_ranking.csv（60 行，3 组，各为 1–20 排列）

# 单元冒烟测试（合成缓存 + 各消融臂 20 步）
.venv/bin/python training/smoke_test.py

# 完整训练（需先跑 data_pipeline/01–07 与 extraction/ 两个提取脚本；GPU）
.venv/bin/python training/train.py --config training/configs/final_all.json \
    --override '{"seed":0,"output_dir":"training/runs/final_all_s0","device":"cuda"}'
```

## 模型说明

`models/deleak_final_s{0,1,2}/best.pt` 与 `models/noleak_final_s{0,1,2}/best.pt` 为最终模型（ListMLE、AdamW lr 3e-4、1500 步/轮 × 10 轮；检查点内含完整配置与监控指标，sha1 见 `code/training/runs-README.md`）。预测为 3 个种子检查点的组内 Borda 集成。预训练双塔全程冻结，未微调——小数据上微调大模型会过拟合，这一选择经消融验证（见 `archive_2026-08-08/docs/plan-v4.md` 与 `code/training/runs-README.md`）。

## 瓶颈认知

### 计算瓶颈

双塔冻结 + 嵌入缓存的代价结构决定了瓶颈分布：

- **训练期**：瓶颈不在 GPU 算力，而在 **CPU 数据供给**——嵌入 npz 的随机读取 + padding 批次上 256 维小矩阵乘的前向，GPU 利用率仅 ~47%。单种子全程 4.5–8.5 分钟（RTX 4090）。
- **推理期**：瓶颈是 **ESM-2 650M 的前向**（CPU 单条约秒级，全长刺突需 3 段滑窗）。
- **端到端**：成本集中在**一次性嵌入提取**（抗体 235,331 条约 6 min / 48 GB fp16；抗原 4,716 条约 2 min / 4.9 GB）；训练与推理本身均为分钟级。这一结构是"3 种子 × 多评估轴 × 十余个消融臂"能在开发周期内完成的前提。

### PLM 的天花板与双塔选型依据

最终两座塔（AntiBERTy 抗体塔 + ESM-2 650M 抗原塔）不是默认选择，而是消融与文献证据收敛的结果（见 `archive_2026-08-08/docs/plan-v4.md` 与 `code/training/runs-README.md`）：

1. **零样本 PLM 打分不可用（地板线）**。亲和力是外在属性（extrinsic property），离开抗原上下文无意义：FLAb/FLAb2 上所有零样本模型的亲和力相关性普遍 ≈0（最优平均 ρ≈0.12，且带显著 germline 偏置）。本方 A0 复测一致（零样本 IgLM/ProGen2 ≈0）。
2. **冻结 PLM 嵌入 + 轻量头是序列骨干的最强形态**。AbAffinity 消融：冻结 ESM-2 骨干 + 轻量头达 Pearson 0.86 / Spearman 0.84；FLAb2 少样本设定下**通用蛋白 LM 嵌入不劣于甚至优于抗体专用嵌入**。这给出两条选型方向：抗体塔取抗体专用 MLM、抗原塔取通用大模型。
3. **抗体塔：AntiBERTy 胜 IgLM（消融 A1）**。双向 MLM 编码器 0.276 ± 0.064 vs 因果 LM 的 IgLM 0.261 ± 0.059，且 IgLM 特征最不稳定（fold0 种子散布 0.094–0.257，三次 epoch-0 早停）——双向上下文的池化表征显著更稳，AntiBERTy 入选。
4. **抗原塔：ESM-2 650M（消融 A2 反证其价值）**。抗原塔不是摆设：去掉抗原塔的盲抗原对照在最严格的抗原聚类留出轴上几乎腰斩（0.083 → 0.054），说明 ESM-2 抗原表征承载了跨抗原泛化的关键信息；650M 的通用蛋白覆盖可处理任意病毒糖蛋白/多肽抗原。
5. **天花板在哪：序列表征已接近饱和**。结构第三塔的四形态消融（ProstT5 / IgFold+SaProt，三种注入粒度）全部落在种子噪声内或为负——因为 IgFold 结构本身是从序列预测的派生产物，整条链路**没有引入序列之外的增量信息**；而组内排序所需的 CDR 细微差异信号，序列嵌入已近乎榨干（残差臂学到近乎恒等映射是机制层面的印证）。换言之，当前框架的瓶颈**不是容量也不是算力，而是冻结序列 PLM 的信息上限**：要进一步突破，需要序列之外的新信息源（真实实验结构、端到端结构微调）或换评测轴。
6. **已知难边界**：整族抗原未见时 Spearman 仅 ~0.08——模型学到的是抗原特异的结合上下文，而非可迁移的抗原生物学。

## 仓库说明（git 版）

本仓库为里程碑快照。为控制体积，以下内容按 `.gitignore` 排除、可按需重建/重下：`.venv/`、`code/training/runs/`（消融与演示检查点，最终模型在 `models/`）、`code/extraction/structures/`（派生缓存）、`planning/`（内部工作笔记）、pandoc 二进制、zip 归档。
