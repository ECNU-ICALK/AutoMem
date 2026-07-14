# 复现说明

[English](reproduction.md)

## 离线检查（无需凭证）

```bash
python -m pip install -e ".[dev,benchmarks]"

automem space
automem smoke
pytest -m "not online"
```

完整跑通两轮合成进化（数据切分、候选、canonical 交接、Pareto 更新、M3 runoff）：

```bash
python -m automem.search.engine \
  --run_name evolution-smoke --output_dir "$(mktemp -d)" \
  --infile examples/smoke_tasks.jsonl \
  --max_rounds 2 --num_candidates 3 \
  --warmup_n 1 --search_n 4 --batch_size 2 --validation_n 1 --test_n 1 \
  --dry_run --no_ledger
```

合成指标只验证控制流，不是基准成绩。

## 数据集

请从官方渠道按各自许可获取基准数据；AutoMem 不会自动下载。输入字段要求：

| Runner | 输入 | 必需字段 |
| --- | --- | --- |
| GAIA | JSON 数组或 JSONL | `Question`、`Final answer`、`file_name`、`task_id`、`Level` |
| WebWalkerQA | JSON 数组或 JSONL | 非空的 `question`、`answer`、`root_url` |
| xBench-DeepSearch | UTF-8 CSV | `id`、`prompt`、`answer`、`reference_steps`、`canary`（base64+XOR-canary 编码；请勿发布解密行） |

GAIA 的 `file_name` 必须为空或限制在输入文件目录内的相对路径；ZIP 附件在严格的
路径/类型/大小限制下解压。所有 runner 的 `--task_indices` 与持久化的 `item_index`
均为一基索引。

## 在线运行

把 `.env.example` 复制为 `.env`，只填你用到的变量——key/base 必须成对配置，且
不要提交任何真实凭证。之后可以单独评测一个基准：

```bash
python -m automem.benchmarks.gaia.runner \
  --infile data/gaia/metadata.jsonl --model TASK_MODEL --judge_model JUDGE_MODEL \
  --memory_provider modular --enable_memory_evolution
```

或启动完整的架构搜索（下面是 GAIA 默认参数；xBench-DeepSearch 默认在其 100 行上
使用 10/70/10/10 切分）：

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

`--model` 只控制任务智能体；`--search_model`、`--diagnosis_model`、`--judge_model`
分别控制各自角色。完整参数用 `python -m automem.search.engine --help` 和各 runner
的 `--help` 查看。
