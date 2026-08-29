# TaiJi MinerU launcher internals

This package contains the shared TaiJi compute lifecycle, Ray Job wrapper,
HDFS/Ceph gates, node-local output spool, GPU monitoring, and SIGSEGV evidence
helpers used by both MinerU engines.

For the frozen 64-GPU reproduction commands and JSON contracts, use
`experiments/mineru_64gpu_repro/README.md`. Do not call this package directly
for a new comparison unless you intentionally need a non-frozen topology or
parameter set.

`profile.toml` is a credential-free example. Generate the ignored
`experiments/mineru_64gpu_repro/profile.local.toml` with the reproduction
frontend before submitting. CMK and PAT token contents must never be committed.
