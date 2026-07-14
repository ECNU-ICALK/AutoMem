# 复现说明

[English](reproduction.md)

## 版本边界

本仓库复现的是 AutoMem 软件及其固定搜索协议，架构合同与论文对齐：Encode 选择五种
抽取类型的非空子集，因此论文中发现的多 Encode 路线（例如 GAIA 上的
`tip+trajectory+workflow`）可以直接表示为严格的 `automem-esrm-v1`
`ArchitectureSpec` 文档。仓库有意不包含基准数据、模型权重、API 凭证、外部 baseline
源码、原始轨迹、记忆池和论文结果文件。

它本身仍然不是 `AutoMAS/paper/main.tex` 的数值复现包：论文实验使用了其自身的数据
快照、数据划分与最终报告协议、模型版本和 prompt 状态。在这些输入尚未另外统一并发布
之前，不能把当前版本跑出的指标标注为论文表格的复现结果。

七月份新增的图边自适应操作已经合并到唯一的 `graph_consolidate` 管理器中，没有第二个
`graph_adaptive` 架构选项。

## 离线检查

完整测试会导入可选的多媒体运行器，因此需要同时安装开发和 benchmark 依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,benchmarks]"

automem space
automem smoke
python -m compileall -q src
ruff check src tests
pytest -m "not online"
```

可以用下面的命令离线运行两个完整的合成进化轮次，覆盖数据划分、候选搜索、canonical
传递、Pareto 更新和 M3 runoff：

```bash
SMOKE_ROOT="$(mktemp -d)"
python -m automem.search.engine \
  --run_name evolution-smoke \
  --output_dir "$SMOKE_ROOT" \
  --infile examples/smoke_tasks.jsonl \
  --max_rounds 2 \
  --num_candidates 3 \
  --warmup_n 1 \
  --search_n 4 \
  --batch_size 2 \
  --validation_n 1 \
  --test_n 1 \
  --dry_run \
  --no_ledger
```

合成指标只验证控制流程，不是 benchmark 结果。

## 数据格式

请从官方来源合法获取数据并遵守相应许可证；AutoMem 不会自动下载数据。

| 运行器 | 输入格式 | 必需字段 |
| --- | --- | --- |
| GAIA | JSON 数组或 JSONL | `Question`、`Final answer`、`file_name`、`task_id`、`Level` |
| WebWalkerQA | JSON 数组或 JSONL | 非空 `question`、`answer`、`root_url`；可选 `info` 必须是对象 |
| xBench-DeepSearch | UTF-8 CSV | `id`、`prompt`、`answer`、`reference_steps`、`canary` |

GAIA 的 `file_name` 必须为空，或是限制在输入文件目录内的相对路径。绝对路径、`..`
越界和符号链接越界都会被拒绝。ZIP 附件只会解压到每个任务独立的临时目录，并检查成员
路径、类型、数量、解压后体积和压缩比。

xBench 的 `prompt` 和 `answer` 应采用该数据集的 base64 加逐行 XOR-canary 编码；不要
公开解密后的题目。三个运行器的 `--task_indices` 和落盘的 `item_index` 都使用 1-based
索引。

## 在线搜索

将 `.env.example` 复制到私有位置，或者只导出实际需要的变量，绝不能提交填入凭证后的
文件。角色专用 endpoint 必须同时配置对应的 key 和 base。先检查当前安装版本的参数：

```bash
python -m automem.benchmarks.gaia.runner --help
python -m automem.benchmarks.webwalkerqa.runner --help
python -m automem.benchmarks.xbench_deepsearch.runner --help
python -m automem.search.engine --help
```

当前版本的一个显式 GAIA 搜索示例是：

```bash
python -m automem.search.engine \
  --run_name gaia-current-v1 \
  --output_dir runs/search \
  --infile data/gaia/metadata.jsonl \
  --benchmark GAIA \
  --model TASK_MODEL_ID \
  --search_model SEARCH_MODEL_ID \
  --judge_model JUDGE_MODEL_ID \
  --diagnosis_model DIAGNOSIS_MODEL_ID \
  --max_rounds 8 \
  --num_candidates 3 \
  --warmup_n 19 \
  --search_n 100 \
  --batch_size 50 \
  --validation_n 30 \
  --test_n 15 \
  --max_steps 40 \
  --token_budget 8192 \
  --concurrency 1 \
  --final_validation
```

`--model` 只控制 benchmark task agent；`--search_model` 控制架构 proposer，
`--diagnosis_model` 和 `--judge_model` 分别保持各自角色。这些是当前发布版参数，不代表
旧论文预算。

当前 xBench 发布默认把 100 题划分为 warmup/search/validation/held-out
`10/70/10/10`。下面是与自动默认等价的显式命令：

```bash
python -m automem.search.engine \
  --run_name xbench-current-v1 \
  --output_dir runs/search \
  --infile data/xbench/deepsearch.csv \
  --benchmark xBench-DeepSearch \
  --model TASK_MODEL_ID \
  --search_model SEARCH_MODEL_ID \
  --judge_model JUDGE_MODEL_ID \
  --diagnosis_model DIAGNOSIS_MODEL_ID \
  --warmup_n 10 \
  --search_n 70 \
  --batch_size 50 \
  --validation_n 10 \
  --test_n 10 \
  --concurrency 1 \
  --final_validation
```

这组 xBench 大小是当前软件协议，不是对旧论文 split 的还原。没有提供 `--data_split` 或
任一 split 大小参数时，搜索引擎会自动采用相同的 10/70/10/10。进化实验必须保持
`--concurrency 1`（搜索
协调器会强制检查）：共享记忆操作虽然有锁，但更高并发下任务完成顺序会改变后续任务能
看到的记忆。standalone runner 允许更高的 shared-memory 并发，但会明确提示记忆累积顺序
不确定，不能把这类运行当作确定性进化复现。

恢复运行时使用完全相同的命令并增加 `--resume`。系统会记录协议摘要，覆盖任务及引用
附件的字节、划分、运行器和包源码、prompt、解析后的模型/endpoint、Web provider/缓存
策略，以及会改变行为的参数。摘要不一致时，依赖旧协议的 checkpoint 会失效。不完整
的状态化候选会从持久化的 round-start canonical 快照整批重跑，不会在已经演化的候选
存储中只补缺失任务。warmup、M3 contender 和最终 memory validation 同样遵守“精确完整
才复用，否则整批重放”；M3 还会让所有 contender 绑定同一个带 manifest 摘要的
`runoff_start_state`。主动改变实验时应使用新的 `--run_name`。

协议摘要相同并不意味着在线证据不可变。受控复现应使用专用 `AUTOMEM_CACHE_DIR`，先填充
一次搜索和页面缓存，保存其哈希或归档，然后设置 `FREEZE_CACHE=true`。否则必须把实时
网页或 provider 漂移记录为限制。缓存内容在正常运行中会增长，因此不会直接纳入协议
摘要。

## 输出与失败语义

运行产物写入指定的 `--output_dir`，并被 Git 忽略。搜索会保存划分、协议签名、无记忆
基线、canonical pool、逐候选任务结果、Pareto 历史、champion 状态、M3 runoff 和可选的
最终留出集报告。

worker 异常、任务基础设施错误、judge 无有效结论、任务结果缺失/多余或 JSON 损坏都会
使运行非零退出，而不会被当成普通答错计分。无效结果不会在 resume 时被跳过；对应候选
会从 round-start 快照重建并整批重试。重试前应检查 `run.log` 和逐任务错误文件。
所有可评分 checkpoint 使用同一个当前合同：文件名是 `<item_index>.json`，内部一基整数
`item_index` 与文件名一致，`status` 必须严格为 `"success"`，`judge_unjudged` 必须显式为
`false`，`task_score` 必须是 `[0,1]` 内有限数。每条结果还必须包含 `task_identity` SHA-256，
它由输入文件的精确 SHA-256 和一基行号确定；runner resume 与 engine 聚合都会和当前数据
核对。自定义 runner 也必须遵守；旧格式、数据身份不符、索引不一致、重复或字段不完整的
checkpoint 都会重跑。
音频检查还需要系统安装 `ffmpeg`，并显式成对配置 `MTU_API_KEY` 和
`MTU_BASE_URL`。

## 复现记录

至少记录 Git commit 和 dirty 状态、Python 与依赖版本、架构 JSON 及 fingerprint、协议和
runtime digest、数据集精确版本与任务 ID、模型/服务版本、endpoint 角色、seed、完整 CLI
参数，以及发布产物的哈希。无法确定的外部模型版本或可变数据集必须明确列为限制。

## Baseline

搜索会自动评估自身的 no-memory 对照。外部 baseline 实现不会放入本仓库；复现者应从
各自官方来源获取，并记录上游 URL、精确 revision、backbone、judge、任务子集和评分
规则。`--baseline_from` 只会在任务索引、模型和 baseline 协议摘要完全一致时复用已有的
AutoMem no-memory baseline。
