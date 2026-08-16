# StEvo-Bench on MiniMax-H3

[StEvo-Bench](https://huggingface.co/datasets/JhanLiufu/StEvo-Bench) ships one
directory per task, holding a YAML spec and the initial frame the video is meant
to evolve from. Each task becomes one first-frame-conditioned generation:
`video_name` is the task `id`, `prompt` is `prompts.video_WM`, `image` is the
initial frame.

## End to end

```bash
# 1. Pull the dataset and convert it into a prompt list. Downloads a snapshot
#    (no .git), writes models/minimax_h3/stevo_bench/{prompts.json,frames/},
#    deletes the download again.
python3 scripts/build_stevo_bench_prompts.py

# 2. Decide the GPU grouping from the topology: NV-linked quads first.
nvidia-smi -L
nvidia-smi topo -m

# 3. Generate. One group of four, or as many disjoint quads as the node has.
python3 scripts/run.py config/minimax_h3/minimax_h3_a100_batch.toml \
  --set H3_PROMPTS_FILE=models/minimax_h3/stevo_bench/prompts.json \
  --set H3_GPU_GROUPS="[0,1,2,3], [4,5,6,7]"
```

More than one group also needs `[slurm].gpus_per_node = 4 x groups` in the
config; `--set` cannot reach a nested key.

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
