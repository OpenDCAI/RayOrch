# Legacy tests

This directory preserves regression cases for deprecated implementations.
It is excluded from the default pytest suite and is not maintained against the
current runtime API.

`overlapped_pipeline/` documents the behavior of the original graphless
`OverlappedPipeline`, which has been superseded by compiled DAG executors.
