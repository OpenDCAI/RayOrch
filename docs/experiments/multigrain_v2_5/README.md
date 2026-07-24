# Multigrain V2.5 实验记录

本目录用于保存可随代码版本追踪的实验方法、参数和结果摘要。

大体积输出、模型权重、Ray 原始 timeline、GPU 采样和生成的 Markdown
不提交到 Git；每份记录会写明这些外部产物的准确路径。

当前记录：

- [`2026-07-24_mineru_overlap_and_batching.md`](2026-07-24_mineru_overlap_and_batching.md)
  - 4×H20 Flash-MinerU 实验；
  - bounded multi-Arena overlap；
  - OCR Dispatch batch 大小归因；
  - Ray timeline、V2.5 Dispatch timeline、GPU/RSS 采样；
  - 与旧 MG 的性能回归对照。
- [`2026-07-24_native_flash_mineru_baseline.md`](2026-07-24_native_flash_mineru_baseline.md)
  - 当前原生 Flash-MinerU commit 的 368-PDF 全量基线；
  - 原生外层 PDF batch 8/24 的 48-PDF 对照；
  - 原生 document-grain OCR 与 V2.5 page-grain elastic batching 的差异；
  - 当前可以成立和仍需进一步 ablation 的性能归因结论。
- [`2026-07-24_parent_bound_vs_elastic_batch_curve.md`](2026-07-24_parent_bound_vs_elastic_batch_curve.md)
  - 同一 V2.5 实现中的 `parent_bound` / `elastic` 严格对照；
  - OCR batch cap 16/32/64/128 的三次 paired repetitions；
  - 实际 RPC size、吞吐、GPU/RSS 与正确性结果；
  - cross-parent elastic rebatching 的净收益。
- [`2026-07-25_parent_bound_vs_elastic_full.md`](2026-07-25_parent_bound_vs_elastic_full.md)
  - 368 PDFs / 7,072 pages 的 4×H20 满载对照；
  - batch cap 64 下三次 paired full runs；
  - 原生 Flash-MinerU、V2.5 parent-bound、V2.5 elastic 三层收益分解；
  - 1,104 份 paired Markdown 正确性对比。

这些文档只记录已经实际运行得到的结果，不据此宣称通用性能、完整的
bounded-memory 性质或论文定理已经成立。

## 可复现 runner

48-PDF batch curve 与 368-PDF 满载 paired 实验可以用同一 runner 执行：

```bash
python -m rayorch.experimental.multigrain_v2_5.benchmark.mineru_ablation run \
  --config docs/experiments/multigrain_v2_5/configs/<config>.json \
  --output-root /path/to/experiment \
  --flash-repo /path/to/Flash-mineru \
  --model /path/to/MinerU2.5-2509-1.2B

python -m rayorch.experimental.multigrain_v2_5.benchmark.mineru_ablation summarize \
  --config docs/experiments/multigrain_v2_5/configs/<config>.json \
  --output-root /path/to/experiment
```

Runner 为每个配置启动独立进程，并支持基于 `done/<tag>` 的断点续跑。
大体积日志、输出和 timeline 仍写到 `output-root`，不进入 Git。
