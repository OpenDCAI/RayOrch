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

这些文档只记录已经实际运行得到的结果，不据此宣称通用性能、完整的
bounded-memory 性质或论文定理已经成立。
