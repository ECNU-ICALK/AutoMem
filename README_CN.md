<div align="center">
  <img src="assets/framework.png" alt="AutoMem 框架图：EGAS（经验引导的架构搜索）与 FGMD（失败引导的模块诊断）" width="900">
  <h1 align="center">AutoMem: Text-Gradient Evolution of LLM-Agent Memory Architectures</h1>
</div>

<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>

## 简介

LLM 智能体的长期记忆设计是一个高度耦合的架构问题：**编码（Encode）**什么、如何
**存储（Store）**、如何**检索（Retrieve）**、如何**管理（Manage）**相互影响，且最优
组合随任务分布变化。AutoMem 把它变成一个在显式分解空间上的搜索问题，用文本梯度驱动
的递归自改进循环求解，由两个组件构成（见上图）：

- **EGAS — 经验引导的架构搜索。** 每轮由 proposer LLM 依据 Pareto 前沿、历史
  rollout 的观测图（Observation Graph）和经验账本（Experience Ledger）提出候选
  架构；文本梯度（瓶颈 → 证据 → 行动）引导下一轮提案，而非盲目变异。
- **FGMD — 失败引导的模块诊断。** 每个候选的失败 rollout 先经范围过滤（剔除工具
  错误、超时、判分歧义），剩余的记忆相关失败被归因到具体模块
  （`extraction_gap`、`retrieval_miss_topk`、`retrieval_noise`、`memory_stale` 等），
  转化为下一轮的定向反馈。

所有架构都运行在代码固定的记忆使用运行时（`automem-runtime-v1`：一次带引用的上下文
组合调用、BEGIN+至多一次刷新的生命周期、保留字面查询的查询规划）之下，搜索不会奖励
隐藏的执行策略差异。详见[架构说明](docs/architecture.md)。

## 架构空间

唯一公开空间 `automem-esrm-v1`——兼容性校验前 31 × 5 × 6 × 4 = 3720 个组合，其中
2573 个合法：

| 维度 | 可选项 |
| --- | --- |
| 编码 Encode（5） | `tip`、`insight`、`trajectory`、`workflow`、`shortcut` |
| 存储 Store（5） | `json`、`vector`、`hybrid`、`graph`、`llm_graph` |
| 检索 Retrieve（6） | `hybrid`、`contrastive`、`cbr_rerank`、`graph`、`hyde`、`mmr` |
| 管理 Manage（4） | `lightweight`、`json_full`、`tool_manager`、`graph_consolidate` |

所有被选中的编码类型写入唯一选定的存储后端。`graph` 检索要求图家族存储；
`graph_consolidate` 额外要求 `graph` 检索器。`tip+trajectory+workflow` 这类多
Encode 路线是一等公民（见 [configs/example.architecture.json](configs/example.architecture.json)）。

## 🚀 快速开始

### 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,benchmarks]"
```

### 2. 离线验证（无需任何 API key）

```bash
automem space          # 打印公开空间与兼容组合数
automem smoke          # 架构 / 存储 / 检索 / 运行时的离线冒烟
pytest -m "not online" # 完整离线测试套件
```

### 3. 跑一次合成进化（离线、零成本）

```bash
python -m automem.search.engine \
  --run_name evolution-smoke --output_dir "$(mktemp -d)" \
  --infile examples/smoke_tasks.jsonl \
  --max_rounds 2 --num_candidates 3 \
  --warmup_n 1 --search_n 4 --batch_size 2 --validation_n 1 --test_n 1 \
  --dry_run --no_ledger
```

### 4. 跑真实架构搜索（需要凭证与数据集）

把 `.env.example` 复制为 `.env` 并填入你使用的服务凭证
（`OPENAI_API_KEY`/`OPENAI_API_BASE`、`SERPER_API_KEY` 等）；从官方渠道获取基准
数据集，然后：

```bash
python -m automem.search.engine \
  --run_name gaia-search --output_dir runs/search \
  --infile data/gaia/metadata.jsonl --benchmark GAIA \
  --model TASK_MODEL --search_model SEARCH_MODEL \
  --judge_model JUDGE_MODEL --diagnosis_model DIAGNOSIS_MODEL \
  --max_rounds 8 --num_candidates 3 \
  --warmup_n 19 --search_n 100 --batch_size 50 --validation_n 30 --test_n 15 \
  --final_validation
```

`--benchmark` 也接受 `WebWalkerQA` 和 `xBench-DeepSearch`。不做搜索、单独评测某个
基准可直接用各 runner，例如：

```bash
python -m automem.benchmarks.gaia.runner \
  --infile data/gaia/metadata.jsonl --model TASK_MODEL --judge_model JUDGE_MODEL \
  --memory_provider modular --enable_memory_evolution
```

数据集字段契约和各基准默认参数见[中文复现说明](docs/reproduction_CN.md)。

## 仓库结构

```text
src/automem/architecture/   公开 schema、兼容规则、编译器
src/automem/providers/      记忆抽取与 provider 生命周期
src/automem/storage/        json / vector / hybrid / 图家族存储
src/automem/retrieval/      检索实现
src/automem/management/     生命周期操作与四个公开预设
src/automem/runtime/        固定的记忆使用执行策略
src/automem/search/         EGAS/FGMD 搜索循环、诊断与选择
src/automem/benchmarks/     GAIA、WebWalkerQA、xBench-DeepSearch 运行器
src/automem/prompts/        随包安装的 prompt 资源
src/flashoagents/           基准运行器使用的改造版智能体运行时
tests/                      离线单元 / 集成 / 冒烟测试
```

仓库只包含源码、prompt 和离线测试——不含数据集、凭证、baseline 或实验结果产物。

## 文档

- [架构说明](docs/architecture.md)——空间、约束、固定运行时
- [配置说明](docs/configuration.md)——`ArchitectureSpec` 契约
- [复现说明](docs/reproduction_CN.md)——运行命令与数据集契约
- [贡献指南](CONTRIBUTING.md) · [安全策略](SECURITY.md)

## 引用与许可证

引用信息见 [CITATION.cff](CITATION.cff)。AutoMem 以 Apache-2.0 许可发布；修改过的
第三方运行时源码保留其文件级版权头（见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）。欢迎通过 GitHub issue 提问。
