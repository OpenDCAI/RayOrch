# Ping-Pong Serving Demo (SGLang ↔ vLLM)

This example demonstrates how two different local serving backends
(SGLang and vLLM) can interact with each other using Ray actors.

Both backends load the same model (e.g. Qwen3-0.6B) but run in
separate conda environments. Prompts are passed back and forth
multiple times ("ping-pong") with different transformations applied
at each step.

---

## What It Shows

- Ray-based multi-actor orchestration
- Separate runtime environments per backend
- Cross-backend prompt passing
- Multi-round prompt transformations
- Transcript logging

---

## Requirements

Prepare two conda environments, one with sglang, one with vllm.

⚠ Ensure:
- Same Ray version in both environments
- Same Python major version
- Same NumPy version

---

## Usage

Edit the environment names and model path:

```python
SGLANG_ENV = "df_sglang_mxc"
VLLM_ENV = "df-vllm-mxc"
MODEL_PATH = "/path/to/Qwen3-0.6B"
```

Then run:

```
python test_serving_pingpong.py
```

---

## Output

* Prints full transcript to console
* Saves results to:

```
./logs/pingpong_demo_<timestamp>.txt
```

---

## Summary

This is a minimal, clean example of:

> Distributed prompt orchestration across heterogeneous serving backends.

