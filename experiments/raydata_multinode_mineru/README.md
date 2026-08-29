# Ray Data MinerU frontend internals

This package retains the Ray Data-specific frontend and result comparison
helpers. Compute lifecycle, input/model validation, output handling, and Ray
Job submission are delegated to `experiments/taiji_multinode_mineru`.

Use `experiments/mineru_64gpu_repro/README.md` and
`experiments/mineru_64gpu_repro/reproduce.py` for the frozen, reproducible
8 x 8 H20 experiment.
