# AutoMem

[English](README.md) | 简体中文

AutoMem 是一个面向大语言模型智能体的任务自适应长期记忆研究框架。它搜索一种由四个
维度明确描述的记忆架构：**编码（Encode）**、**存储（Store）**、**检索（Retrieve）**
和**管理（Manage）**。本仓库包含架构模型、存储与检索实现、管理流水线、固定的记忆使用
运行时、搜索循环和基准测试运行器。本仓库不内置 baseline 仓库、基准数据集、模型权重或
实验结果产物。

架构合同与论文一致：Encode 从五种抽取类型中选择一个非空子集（`tip+trajectory+workflow`
这类多 Encode 路线是一等公民），Store、Retrieve、Manage 各选一个值。本仓库仍然不是
独立的数值复现包——数据集、凭证和实验结果产物均有意排除。比较任何结果前请先阅读
[中文复现说明](docs/reproduction_CN.md)。

## 公开架构空间

唯一的公开架构空间是 `automem-esrm-v1`：

| 维度 | 可选项 |
| --- | --- |
| 编码 Encode（5 选非空子集 → 31 种） | `tip`、`insight`、`trajectory`、`workflow`、`shortcut` |
| 存储 Store（5） | `json`、`vector`、`hybrid`、`graph`、`llm_graph` |
| 检索 Retrieve（6） | `hybrid`、`contrastive`、`cbr_rerank`、`graph`、`hyde`、`mmr` |
| 管理 Manage（4） | `lightweight`、`json_full`、`tool_manager`、`graph_consolidate` |

兼容性校验前，这是一个包含 3720 个组合的空间（31 个 Encode 子集 × 5 × 6 × 4）。所有
被选中的抽取类型都会写入唯一选定的那个存储后端。`graph` 检索要求使用 `graph`
或 `llm_graph` 存储。`graph_consolidate` 将图内容整合与基于成功结果感知的边自适应合并
在同一个管理器中，因此必须同时使用 `graph`/`llm_graph` 图家族存储和 `graph` 检索器。
当前校验器接受其中 2573 个兼容组合。

每个 `ArchitectureSpec` 的 Encode 维度选择一个非空子集（单个字符串等价于只含一个类型
的子集），其余三个维度各选择一个值。缺失值、未知值、重复值和不兼容组合都会被拒绝。
执行行为不是第五个搜索维度。

## 固定运行时

所有架构都使用代码中固定定义的 `automem-runtime-v1` 策略：

- **G2 上下文组合：** 检索到的候选记忆经过一次相关性判断与组合调用，生成暂定且带引用
  的指导信息。如果模型不可用或输出不可用，则使用排名第一的检索结果作为离线兜底。
- **G3 刷新生命周期：** 系统在任务 `BEGIN` 阶段考虑记忆，之后最多只允许在摘要或重新
  规划边界显式请求一次刷新。普通中间步骤不能刷新记忆。
- **G4 查询规划：** 始终保留任务的原始字面查询。抽象查询可以补充语义表示，但不能替换
  原始查询。

这些行为及其限制实现在 `src/automem/runtime/` 中。它们不会出现在
`ArchitectureSpec`、配置文件、环境变量或搜索提案中。

## 仓库结构

```text
src/automem/architecture/   公开架构规范、兼容性规则和编译器
src/automem/providers/      记忆提取和 provider 生命周期
src/automem/storage/        JSON、向量、混合及图存储
src/automem/retrieval/      检索实现
src/automem/management/     生命周期操作和四个公开管理预设
src/automem/runtime/        固定的 G2/G3/G4 执行策略
src/automem/search/         架构搜索、诊断和选择
src/automem/evaluation/     离线指标汇总和基准测试工具
src/automem/benchmarks/     GAIA、WebWalkerQA 和 xBench 运行器
src/automem/prompts/        随 Python 包安装的 prompt 资源
src/flashoagents/           基准运行器使用的修改版智能体运行时
tests/                      离线单元测试、集成测试和 smoke test
docs/                       架构、配置和复现文档
configs/                    可纳入版本控制的公开架构输入规范
examples/                   用于离线控制流程 smoke test 的合成输入
```

运行时数据应放在已被 Git 忽略的 `data/`、`runs/`、`storage/` 等目录中，或者放入外部
产物存储系统。

仓库提供了一个可直接校验的严格配置示例：
[`configs/example.architecture.json`](configs/example.architecture.json)。

## 安装与检查

AutoMem 要求 Python 3.10 或更高版本。默认检查完全离线，不需要 API 凭证：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,benchmarks]"

automem space
automem smoke
pytest -m "not online"
```

依赖组 `benchmarks` 会安装运行三个基准测试及其离线测试所需的向量、Web、文档和媒体
处理依赖。
较轻量的 `web` 依赖组用于 Web 服务适配器，`vector` 依赖组用于仅作为 Python 库使用时
启用向量存储：

```bash
python -m pip install -e ".[benchmarks]"
```

已经实现的运行器和搜索命令请参阅[中文复现说明](docs/reproduction_CN.md)。在线运行还需要自行
合法获取数据集和服务凭证，并明确审查调用成本与数据处理方式。

## 文档

- [架构说明](docs/architecture.md)
- [配置说明](docs/configuration.md)
- [架构配置示例](configs/example.architecture.json)
- [复现说明（中文）](docs/reproduction_CN.md)
- [Reproduction (English)](docs/reproduction.md)
- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)

## 引用与许可证

软件引用信息位于 [CITATION.cff](CITATION.cff)。AutoMem 使用 Apache License 2.0。
修改后的第三方源码保留了各文件原有的版权和许可证头；来源记录的限制说明位于
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
