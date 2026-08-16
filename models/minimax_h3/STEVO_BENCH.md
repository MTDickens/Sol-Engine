# StEvo-Bench on MiniMax-H3

[StEvo-Bench](https://huggingface.co/datasets/JhanLiufu/StEvo-Bench) ships one
directory per task, holding a YAML spec and the initial frame the video is meant
to evolve from. Each task becomes one first-frame-conditioned generation:
`video_name` is the task `id`, `prompt` is `prompts.video_WM`, `image` is the
initial frame.

## End to end

From a clean box to running videos. Steps 1-3 are one-time.

```bash
# 1. Repository. The default branch is main; this runtime lives on sol-engine.
git clone https://github.com/MTDickens/Sol-Engine.git
cd Sol-Engine
git switch sol-engine

# 2. Host environment. Only the launcher and the prompt builder run here -- the
#    GPU stack lives in the pinned SGLang image -- so this stays small.
uv venv --python 3.12
source .venv/bin/activate
uv pip install huggingface_hub pyyaml

# 3. Weights: the released BF16 FL2VA checkpoint, 269 GiB. HF_HOME must point at
#    a disk that has room for it.
export HF_HOME=/large/disk/hf
hf auth login                      # only if the repo is gated for you
hf download MiniMaxAI/MiniMax-H3

# 4. Pull StEvo-Bench and convert it into a prompt list. Downloads a snapshot
#    (no .git), writes models/minimax_h3/stevo_bench/{prompts.json,frames/},
#    deletes the download again.
python3 scripts/build_stevo_bench_prompts.py

# 5. Decide the GPU grouping from the topology. With NVSwitch every pair is
#    NV-linked, so group along the CPU socket boundary instead.
nvidia-smi -L
nvidia-smi topo -m

# 6. Generate. One group of four, or as many disjoint quads as the node has.
python3 scripts/run.py config/minimax_h3/minimax_h3_a100_batch.toml \
  --set H3_PROMPTS_FILE=models/minimax_h3/stevo_bench/prompts.json \
  --set H3_GPU_GROUPS="[0,1,2,3], [4,5,6,7]" \
  --set H3_CONTAINER_RUNTIME=apptainer \
  --set H3_MODEL_PATH=/large/disk/hf/hub/models--MiniMaxAI--MiniMax-H3/snapshots/<rev>
```

`H3_CONTAINER_RUNTIME` follows the machine, not the run, and the config defaults
to the Slurm one:

- **Bare node** (a Brev instance, a workstation): `apptainer` or `singularity`,
  which makes the launcher pull the pinned image itself. `none` runs in the
  ambient environment instead, and only works if that environment *is* the
  pinned SGLang build -- the runtime hard-checks torch `2.11.0+cu130` and Triton
  `3.6.0` and refuses anything else. The venv in step 2 is not it.
- **Slurm with pyxis:** the config default, nothing to set. Two groups then also
  need `[slurm].gpus_per_node = 8`, which is a nested key that has to be edited
  in the config -- `--set` cannot reach it.

Either way the whole run is one container: the launcher fans the groups out as
processes with `CUDA_VISIBLE_DEVICES` pinned, not as separate jobs.

For a container run every path involved -- the repository, the prompt list,
`frames/`, the cache, the output directory, and a local checkpoint -- has to sit
below `H3_STORAGE_ROOT`, which is mounted at `/h3`. It defaults to the
repository root, so put the clone and `HF_HOME` on the same large disk, or set
`--set H3_STORAGE_ROOT=/large/disk` and keep everything under it. Drop
`H3_MODEL_PATH` entirely to pull the weights from the Hub inside the container.

## Selecting task sets

`--tasks` takes `image_implied`, `simple_trigger`, both (the default), or
neither. Only the selected sets are downloaded.

```bash
python3 scripts/build_stevo_bench_prompts.py --tasks image_implied
python3 scripts/build_stevo_bench_prompts.py --tasks           # writes an empty list, downloads nothing
```

Other flags: `--out-dir` (default `models/minimax_h3/stevo_bench`), `--repo-id`,
`--revision`, and `--keep-snapshot DIR` to leave the download in place.

## Output

```
models/minimax_h3/stevo_bench/
├── prompts.json      one item per task
└── frames/<id>.png   the initial frames, ~1.6 MB each
```

`frames/` is rebuilt on every run, so narrowing `--tasks` cannot leave frames
behind that the list no longer references. Image paths inside `prompts.json` are
relative to it, which is what makes the directory portable into the container --
the launcher resolves them under `/h3`. A task is skipped, and named at the end
of the run, if its `prompts.video_WM` is empty, its frame is missing, or its `id`
collides with one already taken.

The full run is 221 tasks (134 `image_implied` + 87 `simple_trigger`) at roughly
61 s each: about 3.7 h on one group of four A100s, divided by the number of
groups.

## Batch mechanics

Everything about how the batch runner works -- prompt list dialects, GPU
grouping, warmup, `batch.json` -- is in
[`A100/README.md`](A100/README.md#batch-generation). One caveat matters here:
an item with an `image` is sent as an `fl2va` request rather than `t2va`, and
the exact task name and condition entry could not be checked against any
existing conditioned request in this tree. Both are overridable with
`H3_FIRST_FRAME_TASK` and `H3_IMAGE_CONDITION_JSON`, and whatever was sent is
recorded per item in `batch.json`.

## Side note: provisioning the eight-GPU node

The two-group invocation above assumes a node with eight A100s. On NVIDIA Brev
one comes from the academic-grant reserved pool, and that pool is invisible to
`brev search` -- search reads the public catalog, while grant-only types are
bound to the organization and only appear in its own available-types list. Name
the type directly instead of searching for it:

```bash
brev set "<grant org>"

# Confirms the type is accepted. Starts nothing, spends no grant hours.
brev create sana-8xa100 --type azurerm.a100x8.sxm.academic-grant --dry-run

brev create sana-8xa100 --type azurerm.a100x8.sxm.academic-grant \
  --startup-script @provision.sh

brev refresh          # only for instances created in the web console
ssh sana-8xa100
```

`ssh <name>` resolves because the CLI maintains `~/.brev/ssh_config` -- one host
entry per instance, a cloudflared `ProxyCommand`, and `IdentityFile
~/.brev/brev.pem` -- and adds an `Include` for that file to `~/.ssh/config`.
Nothing reads `~/.brev/` on its own. `ssh -G sana-8xa100` prints the resolved
configuration without connecting, which is the way to tell an `Include` ordering
problem from a dead instance.

Three things decide whether the batch run above works on the node it lands on:

- The default SSH target is the instance's container, not the GPU VM. The batch
  run needs the host: eight visible devices, the pinned SGLang image, and the
  large disk. Use `brev shell sana-8xa100 --host`, and read `nvidia-smi -L`
  there before trusting an `H3_GPU_GROUPS` value.
- On this instance type `[0,1,2,3], [4,5,6,7]` is also the CPU socket boundary,
  so it is the split `nvidia-smi topo -m` in step 2 should agree with. The 96
  vCPUs divide into the `cpus_per_task = 48` each group already asks for.
- Grant hours bill from the moment the instance is running, so the image pull
  and the 269 GiB checkpoint pull belong in `--startup-script` rather than an
  interactive session. `H3_STORAGE_ROOT` has to name the large disk, since the
  prompt list, `frames/`, and the output directory all have to sit below it for
  a container run to resolve them.
