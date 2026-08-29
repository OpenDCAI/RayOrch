# MinerU 64-GPU RayOrch / Ray Data reproduction

This directory is the canonical entry point for reproducing the two completed
MinerU runs on TaiJi WeData. Both runs use the same frozen 3,690-PDF input,
model, 8 x 8 Zhongwei H20 compute, actor layout, batch size, image, and output
protocol. The only intended experimental variable is the execution engine.

The complete machine-readable contracts are:

- `configs/rayorch_64gpu.json`
- `configs/raydata_64gpu.json`

The launcher never stores CMK contents or the TaiJi PAT token in Git or in its
JSON state. It copies the PAT token into the temporary Ray working-directory
snapshot only for a Ceph-output submission.

## Frozen environment

- TaiJi application group: `TaiJi_HYAide_LLM_Pretrain_Data`
- Region and accelerator: Zhongwei (`zw`), H20
- Compute: 8 workers x 8 GPUs = 64 GPUs
- TaiJi image: `ray-vllm:2.51.1-py312-tlinux-cu122-vllm11`
- Ray 2.51.1, vLLM 0.11.0, Torch 2.8.0+cu128, PyArrow 19.0.1
- `hydp-engine==0.1.23`, `hydp-dataflow-wedata==0.1.0`
- Flash-MinerU commit: `7246a353554cee35674d7ef82f2c061db2f43003`

The Flash-MinerU repository must have no local changes below `flash_mineru/`.
The launcher checks both the full commit SHA and that subtree before planning
or submitting.

## 1. Prepare the checkout

```bash
git checkout codex/mineru-raydata-rayorch-repro

git clone ssh://git@ssh.github.com:443/OpenDCAI/Flash-mineru.git /path/to/Flash-mineru
git -C /path/to/Flash-mineru checkout 7246a353554cee35674d7ef82f2c061db2f43003
```

Run the offline contract check:

```bash
python3 experiments/mineru_64gpu_repro/reproduce.py verify \
  --flash-repo /path/to/Flash-mineru
```

## 2. Create a local TaiJi profile

The CMK and PAT token files must already exist and remain outside Git. Generate
the ignored local profile using your own CMK file:

```bash
python3 experiments/mineru_64gpu_repro/reproduce.py configure \
  --cmk-file /absolute/path/to/your/cmk
```

This writes `experiments/mineru_64gpu_repro/profile.local.toml`. The file only
contains the CMK path, not the CMK value, and is ignored by Git.

The submitting environment must be able to import `hydp_engine` and
`hydp_dataflow_wedata`. It must also have permission to read the frozen HDFS
inputs/model and request H20 resources from the application group above.

## 3. Render credential-free plans

Always inspect `plan` output before a submission. Pick a globally unique run
ID; the output path must end in that run ID.

RayOrch:

```bash
python3 experiments/mineru_64gpu_repro/reproduce.py plan \
  --engine rayorch \
  --run-id mineru-rayorch-64gpu-YYYYMMDD-HHMMSS \
  --flash-repo /path/to/Flash-mineru
```

Ray Data:

```bash
python3 experiments/mineru_64gpu_repro/reproduce.py plan \
  --engine raydata \
  --run-id mineru-raydata-64gpu-YYYYMMDD-HHMMSS \
  --flash-repo /path/to/Flash-mineru
```

The default output root is:

```text
/apdcephfs_zwfy10/share_304380933/hunyuan/clapliang/rayorch_test
```

Use `--output-root /your/authorized/zw/ceph/root` to select another authorized
Zhongwei Ceph root. The generated output is `<OUTPUT_ROOT>/<RUN_ID>`.

## 4. Submit

RayOrch:

```bash
python3 experiments/mineru_64gpu_repro/reproduce.py submit \
  --engine rayorch \
  --run-id mineru-rayorch-64gpu-YYYYMMDD-HHMMSS \
  --flash-repo /path/to/Flash-mineru \
  --ceph-token-file /absolute/path/to/your/taijiPATToken
```

Ray Data:

```bash
python3 experiments/mineru_64gpu_repro/reproduce.py submit \
  --engine raydata \
  --run-id mineru-raydata-64gpu-YYYYMMDD-HHMMSS \
  --flash-repo /path/to/Flash-mineru \
  --ceph-token-file /absolute/path/to/your/taijiPATToken
```

Each submission creates its own 8 x 8 compute unless
`--reuse-state-file <TERMINAL_OWNED_COMPUTE_STATE>` is explicitly provided.
The orchestrator runs a read-only topology/dependency/HDFS probe before the
full job. It detaches only after two consecutive `RUNNING` observations and
keeps the compute for inspection.

State and driver logs are written below
`experiments/mineru_64gpu_repro/state/` and are intentionally ignored by Git.
Use the shared lifecycle commands for inspection and cleanup:

```bash
python3 experiments/taiji_multinode_mineru/orchestrate.py status \
  --state-file experiments/mineru_64gpu_repro/state/<RUN_ID>.json

python3 experiments/taiji_multinode_mineru/orchestrate.py cleanup \
  --state-file experiments/mineru_64gpu_repro/state/<RUN_ID>.json \
  --profile experiments/mineru_64gpu_repro/profile.local.toml:ray-h20-8x8-zw-pretrain \
  --confirm-compute-id <COMPUTE_ID>
```

Cleanup requires the exact compute ID recorded in the state file and never
deletes HDFS input or produced documents.

## Reproduction invariants

The launcher rejects a run if any of these invariants change:

- compute is not exactly 8 workers x 8 H20 GPUs;
- the application group or region changes;
- OCR reservations do not equal 64 GPUs (`128 x 0.5`);
- the two frozen manifests do not total 3,690 PDFs;
- Flash-MinerU is not at the pinned revision or its package subtree is dirty;
- the output escapes the explicitly authorized Ceph root;
- an existing state file would be overwritten.

The Ray Data engine does not consume `microbatch_size` or
`max_active_microbatches`; those values remain in the shared launcher contract
for parity. Ray Data's effective knobs are explicitly separated in its JSON.
