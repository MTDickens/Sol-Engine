# MiniMax-H3 on A100

## Overview

This self-contained runtime runs the released BF16 FL2VA checkpoint on four A100 GPUs. It uses
SGLang FSDP inference with Ulysses-4, keeps the model resident without offload, and does not modify
the installed SGLang checkout.

## Performance

| GPUs | Workload | Baseline (s) | Optimized (s) | Speedup |
|---:|---|---:|---:|---:|
| 4 | 768p@5s | 217.32 | 61.28 | **3.55x** |

The speedup is measured against the matching baseline runtime. The released configuration is pinned
by [`minimax_h3_a100_fullopt.toml`](../../../config/minimax_h3/minimax_h3_a100_fullopt.toml).

## Full-Opt

- **Parallelism:** FSDP inference with Ulysses-4 and no model offload.
- **Attention:** Triton Sol-Attn with `tau=1.0`, exact thresholding, a full-prefix KV sink, dense
  prefix queries, and the first 10 steps and first two blocks dense.
- **Cache:** FirstBlockCache with threshold `0.08` and synchronized decisions across ranks.
- **Runtime:** pinned SGLang BF16 execution without `torch.compile` or token reordering.

## Usage

One command runs any arm. It reproduces the [`demo_prompt`](../demo_prompt.json)
benchmark with seed `0`, 50 denoising steps and the workload above. Run it from
the repository root:

```bash
python3 scripts/run.py config/minimax_h3/minimax_h3_a100_fullopt.toml                 # the optimized arm
python3 scripts/run.py config/minimax_h3/a100_dense.toml                 # the control it is measured against
```

`scripts/run.py` takes either config dialect -- a flat single-file config or a
config manifest -- and renders the same run bundle under `runs/`:
`launch.sh`, `job.sbatch`, `manifest.resolved.toml`, `metadata.json` and
`outputs/`. Add `--print` to resolve without running, or `--set KEY=VALUE` to
override one value for a single run without editing the config:

```bash
python3 scripts/run.py config/minimax_h3/minimax_h3_a100_fullopt.toml \
  --set H3_STORAGE_ROOT=/shared/path/Sana
```

It contains no scheduler. To run under Slurm either put that same command in
your own job script, or call the renderer directly, which is the one thing
`run.py` does not do:

```bash
python3 scripts/launch_config.py config/minimax_h3/minimax_h3_a100_fullopt.toml --mode sbatch --confirm-submit
```

## Batch Generation

`run_minimax_h3_gpu.sh` generates the one prompt a benchmark number needs.
[`run_minimax_h3_batch.sh`](run_minimax_h3_batch.sh) generates a whole prompt
list on the same profile, loading the checkpoint once per GPU group instead of
once per prompt:

```bash
python3 scripts/run.py config/minimax_h3/minimax_h3_a100_batch.toml
python3 scripts/run.py config/minimax_h3/minimax_h3_a100_batch.toml \
  --set H3_PROMPTS_FILE=models/minimax_h3/my_prompts.json
```

Running the StEvo-Bench tasks through this is written up separately:
[`STEVO_BENCH.md`](../STEVO_BENCH.md).

- **Prompt list:** `H3_PROMPTS_FILE`, JSON, modelled on
  [`demo_prompts.json`](../demo_prompts.json). Each item is
  `{"video_name": ..., "prompt": ..., "seed": optional, "image": optional}`, or
  the shorter `[prompt, video_name]` pair. `video_name` must be unique; `seed`
  defaults to `H3_SEED`. The file must live below `H3_STORAGE_ROOT` for
  container runs.
- **First frame:** `image` on an item makes it an image-conditioned request
  instead of `t2va`. The path is resolved relative to the prompt list, so a
  container run finds it under `/h3` with no host path baked into the JSON, and
  a missing file fails before the checkpoint is loaded. A list may mix
  conditioned and text-only items.

  No runtime in this tree issued a conditioned request before, so the task name
  and the condition entry are the one part of this path that was not checked
  against a working example -- they come from SGLang's sampling params inside
  the pinned image. Both are overridable so a correction is a config change:
  `H3_FIRST_FRAME_TASK` (default `fl2va`) and `H3_IMAGE_CONDITION_JSON`, a JSON
  condition entry whose every `{image}` is replaced with the resolved path, for
  example `'{"type":"image","role":"first_frame","path":"{image}"}'`. Whatever
  was sent is recorded per item in `batch.json`.
- **Outputs:** one `outputs/<video_name>.mp4` per item, plus `batch.json` with
  per-item timings, cache reuse and sparse-rank coverage, and one
  `outputs/group<N>.log` per group.
- **Parallelism:** `H3_GPU_GROUPS` defaults to `[0,1,2,3]`, one group. Give it
  more groups -- `"[0,1,2,3], [4,5,6,7]"` or `"(0,1,2,3), (4,5,6,7)"` -- and the
  prompt list is split into even contiguous shards, one process per group with
  `CUDA_VISIBLE_DEVICES` pinned to it. Every group must be exactly
  `H3_GPUS_PER_GROUP` wide, which must equal `[official_config].num_gpus`;
  groups may not overlap. Under Slurm the ids index the devices the job was
  given, so raise `[slurm].gpus_per_node` to `4 x groups` in the config -- it is
  a nested key that `--set` cannot reach.
- **Warmup:** off by default, since a warmup request is a full generation that
  gets deleted. `H3_BATCH_WARMUP=1` restores it, which makes the per-item
  timings comparable to `benchmark.json` and brings back route-density
  telemetry.

Sampling settings come from `gpu_infer.py` unchanged, so a batch video is the
same 768p 16:9, 124-frame, 50-step generation as the benchmarked one. A prompt
that fails is recorded in `batch.json` and the rest of the shard continues; the
run then exits non-zero.

## Environment

- **Runtime:** `lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86`, PyTorch 2.11.0, and Triton 3.6.
- **Weights:** released BF16 FL2VA checkpoint; set `H3_MODEL_PATH` for an offline local copy.
- **Placement:** four A100 GPUs with shared access to `H3_STORAGE_ROOT`.

The launcher supports Pyxis, Apptainer/Singularity, and native execution. Site-specific Slurm account
and partition settings remain external to the config.

## Outputs

The run bundle stores `out.mp4`, `benchmark.json`, and launch logs under `runs/`.
