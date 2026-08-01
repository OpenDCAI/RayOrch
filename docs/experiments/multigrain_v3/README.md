# Multigrain V3 experiments

本目录只提交实验设计、可复现 runner 和精简结果。模型输出、Ray timeline、GPU samples
和完整 Markdown 应写入 `/tmp` 或用户指定的未跟踪目录。

## 当前 runner

### MinerU V3

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
python -m rayorch.experimental.multigrain_v3.benchmark.mineru ...
```

### MinerU 裸 Ray Data

无 GPU 时先验证 columnar transport：

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
python -m rayorch.experimental.multigrain_v3.benchmark.mineru_ray_data \
  --smoke-no-model \
  --limit 4 \
  --output-dir /tmp/mgv3-mineru-rd-out \
  --artifact-dir /tmp/mgv3-mineru-rd-artifacts \
  --result-jsonl /tmp/mgv3-mineru-rd.jsonl
```

GPU 可见后移除 `--smoke-no-model`，按 4 → 48 → 368 PDFs 递进。

### MinerU native 与 full matrix

Flash-MinerU 当前 pipeline-parallel native baseline：

```bash
python -m rayorch.experimental.multigrain_v3.benchmark.mineru_native \
  --limit 48 \
  --replicas 4 \
  --batch-size 24 \
  --inflight 3 \
  --output-dir /tmp/mineru-native/out \
  --artifact-dir /tmp/mineru-native/artifacts \
  --result-jsonl /tmp/mineru-results.jsonl
```

native runner 在开始 `measured_wall` 前显式等待 render/OCR/convert actors 的
`__ray_ready__`，因此模型/actor startup 与 measured pipeline wall 分开。

生成四模式 × 三次重复的顺序命令清单：

```bash
python -m rayorch.experimental.multigrain_v3.benchmark.mineru_matrix \
  --output-root /path/to/mineru-matrix \
  --limit 368 \
  --repeats 3 \
  > /path/to/mineru-matrix/commands.json
```

清单包含：

```text
V3 elastic
V3 parent_bound
Ray Data
Flash-MinerU native DAG
```

命令应顺序执行，不能让多个模式同时抢占相同 GPU。
不同 repeat 会反转四种 engine 的执行顺序，减少固定顺序带来的文件缓存、热状态和温度偏差。

全部命令完成后，先校验 matrix 完整性再生成中位数：

```bash
python -m rayorch.experimental.multigrain_v3.benchmark.mineru_report \
  --results-jsonl /path/to/mineru-matrix/results.jsonl \
  --expected-repeats 3 \
  --expected-pdfs 368 \
  --output /path/to/mineru-matrix/summary.json
```

四模式主比较使用 `end_to_end_wall_s`（包含 actor/model startup），因为 Ray Data 无公开、
稳定的 actor readiness barrier。报告同时保留各 runner 的 `measured_wall_s` 作为诊断，但
不能把 V3/native 的 post-readiness wall 与 Ray Data startup-inclusive wall 混作公平速度对比。

比较任意两个 engine 的 Markdown correctness：

```bash
python -m rayorch.experimental.multigrain_v3.benchmark.mineru_correctness \
  --left /path/to/v3/outputs \
  --right /path/to/ray-data-or-native/outputs \
  --output /path/to/correctness.json
```

### Video V3 vs Ray Data

只下载 UCF101 subset 中两个小 AVI：

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
RAY_DATA_DISABLE_PROGRESS_BARS=1 \
python -m rayorch.experimental.multigrain_v3.benchmark.video.compare \
  --dataset ucf101 \
  --stride 10 \
  --max-frames 16 \
  --transform-backend resnet18
```

调度消融可把两个公开 AVI 重复为多个独立 logical videos；这不会增加数据下载：

```bash
... --repeat-inputs 8 --batch-scope elastic
... --repeat-inputs 8 --batch-scope parent_bound
```

正式记录前至少加入：

```bash
... --warmup 1 --repeats 3
```

CLI 会输出各次 wall、median 和 paired speedup；不要只报告一次运行。

每个 trial 都会重新创建 V3/ Ray Data actors，因此 wall 是 startup-inclusive。paired trial
交替使用 `V3 first` 和 `Ray Data first`，降低 OS page cache/执行顺序偏差；这里的 warmup
只预热模型文件和系统缓存，不代表 actor 复用后的 steady-state latency。

### Docling V3 vs Ray Data vs native

Docling 依赖建议装在隔离环境。当前容器没有 `libGL.so.1`，因此需要 headless OpenCV：

```bash
python -m venv --system-site-packages /tmp/mgv3-docling-env
/tmp/mgv3-docling-env/bin/python -m pip install docling opencv-python-headless
/tmp/mgv3-docling-env/bin/python -m pip uninstall -y opencv-python
```

如果卸载 GUI wheel 同时删除了共享 `cv2` 文件，再强制恢复 headless wheel：

```bash
/tmp/mgv3-docling-env/bin/python -m pip install \
  --force-reinstall --no-deps opencv-python-headless
```

运行：

```bash
PYTHONPATH=/tmp/mgv3-docling-env/lib/python3.12/site-packages:$PYTHONPATH \
DOCLING_DEVICE=cpu \
HF_HOME=/tmp/mgv3-docling-hf \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
RAY_DATA_DISABLE_PROGRESS_BARS=1 \
python -m \
  rayorch.experimental.multigrain_v3.benchmark.document_docling.compare \
  --paths /path/to/input.pdf \
  --page-batch-size 4 \
  --page-replicas 1
```

`--skip-native` 可跳过整 PDF Docling baseline。

## 测试

快速、无模型测试：

```bash
python -m pytest -q test/experimental/multigrain_v3/benchmark
```

包含本地 Ray 的完整 V3 回归：

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
RAY_DATA_DISABLE_PROGRESS_BARS=1 \
python -m pytest -q test/experimental/multigrain_v3
```

正式实验矩阵、指标和当前证据见：

```text
experiment_matrix.md
2026-08-01_diversity_feasibility.md
2026-08-01_full_system_comparison.md
```
