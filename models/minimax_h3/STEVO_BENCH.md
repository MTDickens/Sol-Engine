# 在 MiniMax-H3 上跑 StEvo-Bench

[StEvo-Bench](https://huggingface.co/datasets/JhanLiufu/StEvo-Bench) 每个任务一个目录，
里面是一份 YAML 规格和该视频要从之演化的初始帧。每个任务对应一次首帧条件生成：
`video_name` 取任务的 `id`，`prompt` 取 `prompts.video_WM`，`image` 取初始帧。

## 一条龙

从一台干净的机器到视频跑起来。第 1-3 步是一次性的。

```bash
# 1. 仓库。远端默认分支是 main，这套 runtime 在 sol-engine 上。
git clone https://github.com/MTDickens/Sol-Engine.git
cd Sol-Engine
git switch sol-engine

# 2. 宿主环境。这里只跑 launcher 和 prompt 构建脚本 —— GPU 那套栈在固定的 SGLang
#    镜像里 —— 所以装这么点就够。
uv venv --python 3.12
source .venv/bin/activate
uv pip install huggingface_hub pyyaml

# 3. 权重：发布的 BF16 FL2VA checkpoint，269 GiB。HF_HOME 必须指向放得下它的盘。
export HF_HOME=/large/disk/hf
hf auth login                      # 仅当该仓库对你是 gated 时才需要
hf download MiniMaxAI/MiniMax-H3

# 4. 拉取 StEvo-Bench 并转成 prompt 列表。下载的是 snapshot（不带 .git），产出
#    models/minimax_h3/stevo_bench/{prompts.json,frames/}，然后把下载删掉。
python3 scripts/build_stevo_bench_prompts.py

# 5. 从拓扑决定 GPU 分组。有 NVSwitch 时任意两卡都是 NVLink 直连，
#    所以改按 CPU socket 边界分组。
nvidia-smi -L
nvidia-smi topo -m

# 6. 生成。一组四卡，或者这台机器能凑出几组不相交的四卡就开几组。
python3 scripts/run.py config/minimax_h3/minimax_h3_a100_batch.toml \
  --set H3_PROMPTS_FILE=models/minimax_h3/stevo_bench/prompts.json \
  --set H3_GPU_GROUPS="[0,1,2,3], [4,5,6,7]" \
  --set H3_CONTAINER_RUNTIME=apptainer \
  --set H3_MODEL_PATH=/large/disk/hf/hub/models--MiniMaxAI--MiniMax-H3/snapshots/<rev>
```

`H3_CONTAINER_RUNTIME` 跟着机器走，而不是跟着这次运行走，而配置里的默认值是给 Slurm 的：

- **裸机**（Brev 实例、工作站）：用 `apptainer` 或 `singularity`，让 launcher 自己去拉
  那个固定镜像。`none` 是直接在当前环境里跑，只有当那个环境**本身就是**那套固定的
  SGLang 构建时才成立 —— runtime 会硬校验 torch `2.11.0+cu130` 和 Triton `3.6.0`，
  对不上就拒绝。第 2 步那个 venv 不是它。
- **Slurm + pyxis**：就是配置的默认值，什么都不用设。但开两组时还需要把
  `[slurm].gpus_per_node` 改成 8 —— 那是个嵌套键，只能改配置文件，`--set` 够不着。

无论哪种方式，整次运行都只在一个容器里：launcher 是把各组作为进程展开、各自钉住
`CUDA_VISIBLE_DEVICES`，而不是拆成多个作业。

容器模式下，牵涉到的每一个路径 —— 仓库、prompt 列表、`frames/`、cache、输出目录、
以及本地 checkpoint —— 都必须位于 `H3_STORAGE_ROOT` 之下，它会被挂载到 `/h3`。
该变量默认是仓库根目录，所以要么把 clone 和 `HF_HOME` 放在同一块大盘上，要么设
`--set H3_STORAGE_ROOT=/large/disk` 并把所有东西都放到它下面。不想管本地权重路径，
就把 `H3_MODEL_PATH` 整行删掉，让容器内直接从 Hub 拉。

## 选择任务集

`--tasks` 接受 `image_implied`、`simple_trigger`、两个都要（默认），或者两个都不要。
只有被选中的任务集才会被下载。

```bash
python3 scripts/build_stevo_bench_prompts.py --tasks image_implied
python3 scripts/build_stevo_bench_prompts.py --tasks           # 写出一份空列表，什么都不下载
```

其余参数：`--out-dir`（默认 `models/minimax_h3/stevo_bench`）、`--repo-id`、
`--revision`，以及 `--keep-snapshot DIR` 用于保留下载的快照。

## 产物

```
models/minimax_h3/stevo_bench/
├── prompts.json      每个任务一条
└── frames/<id>.png   初始帧，每张约 1.6 MB
```

`frames/` 每次运行都会重建，所以缩小 `--tasks` 不会留下列表里已经不再引用的帧。
`prompts.json` 里的图像路径是相对它自身的，正是这一点让整个目录可以原样搬进容器 ——
launcher 会在 `/h3` 下解析它们。若某个任务的 `prompts.video_WM` 为空、初始帧缺失，
或者 `id` 与已有的撞车，该任务会被跳过，并在运行结束时列出来。

跑全量是 221 个任务（134 个 `image_implied` + 87 个 `simple_trigger`），每个约 61 秒：
单组四卡 A100 约 3.7 小时，按组数等分。

## 批量运行机制

批量 runner 的一切 —— prompt 列表的几种写法、GPU 分组、warmup、`batch.json` ——
都在 [`A100/README.md`](A100/README.md#batch-generation)。这里只强调一条：带 `image`
的条目会以 `fl2va` 请求发出，而不是 `t2va`，而具体的 task 名和 condition 条目的形状
在本仓库里找不到任何一个已有的条件请求可供核对。两者都可以用
`H3_FIRST_FRAME_TASK` 和 `H3_IMAGE_CONDITION_JSON` 覆盖，实际发出去的内容会逐条
记录在 `batch.json` 里。

## 可选：把 checkpoint 和产物放进内存

这台八卡实例有 1.9 TiB 内存，而它底下那块 Azure SSD 是整台机器上最慢的东西。
269 GiB 的 checkpoint 在模型加载时每个 GPU 组要读一遍，而它其实完全不必碰那块盘：
直接下载进 tmpfs，之后每次都从内存读，一次都不走 SSD。挂两个，一个放权重，一个放
run bundle：

```bash
# 放在 provision.sh 里，下载之前执行。
mkdir -p /mnt/ram/hf /mnt/ram/runs
mount -t tmpfs -o size=300G,mode=1777 tmpfs /mnt/ram/hf
mount -t tmpfs -o size=300G,mode=1777 tmpfs /mnt/ram/runs
```

`size=` 是上限而不是预留 —— tmpfs 是有人写才分配一页 —— 所以第二个挂载在用起来之前
不花一分钱，而 221 个视频也就个位数 GB。真正花预算的是 checkpoint 那个挂载，而且它
对整台机器只花一次：两个 GPU 组读的是同一份 tmpfs，所以 269 GiB 只从内存里扣一次，
剩下的由两个加载进程分。

这样一来 storage root 就必须是这两个挂载的父目录，clone 也得跟着挪到它下面。
第 1-2 步除了路径以外没有变化，第 3 步和第 6 步变成：

```bash
git clone https://github.com/MTDickens/Sol-Engine.git /mnt/ram/Sol-Engine

export HF_HOME=/mnt/ram/hf
hf download MiniMaxAI/MiniMax-H3

python3 scripts/run.py config/minimax_h3/minimax_h3_a100_batch.toml \
  --run-root /mnt/ram/runs \
  --set H3_STORAGE_ROOT=/mnt/ram \
  --set H3_MODEL_PATH=/mnt/ram/hf/hub/models--MiniMaxAI--MiniMax-H3/snapshots/<rev> \
  --set H3_PROMPTS_FILE=models/minimax_h3/stevo_bench/prompts.json \
  --set H3_GPU_GROUPS="[0,1,2,3], [4,5,6,7]" \
  --set H3_CONTAINER_RUNTIME=apptainer
```

`--run-root` 接受绝对路径，会把整个 bundle —— `launch.sh`、`manifest.resolved.toml`、
`outputs/` —— 放到第二个挂载上。`H3_STORAGE_ROOT` 必须是 `/mnt/ram` 而不是其中任何
一个挂载，因为仓库、prompt 列表、输出目录和本地 checkpoint 全都得在它下面，否则
launcher 会直接拒绝这次运行。

给这次运行设 `HF_HOME` 是没用的。容器模式下 launcher 会在把路径重映射进 `/h3` 之后
从 `H3_CACHE_ROOT` 推导出它，宿主上的值会被覆盖；它只对在容器外执行的 `hf download`
有意义。真正把运行指向 tmpfs 那份副本的是 `H3_MODEL_PATH`。

在投入 3.7 小时之前有两件事要确认：

- **容器必须能看见嵌套挂载。** 对 `/mnt/ram` 做 bind 时，只有递归绑定才会把它下面那两个
  tmpfs 一起带进去，而 launcher 只绑一个路径。一条命令就能确定：
  `apptainer exec --bind /mnt/ram:/h3 <image> df -h /h3/hf /h3/runs` —— 两行都显示
  `tmpfs` 才是想要的结果；两行都显示父文件系统，说明容器读到的是两个空目录。
  换成在 `/mnt/ram` 本身挂一个 tmpfs、把 `hf/` 和 `runs/` 当作普通子目录，可以绕开这个
  问题，代价是失去两个独立的上限。
- **Swap。** `swapon --show`。如果这台实例有 swap，tmpfs 的页可能被换回到我们本想绕开的
  那块 SSD 上。

两个挂载都扛不过重启或 `brev stop`。丢掉 checkpoint 的代价是重新下载一次 —— 这是与
grant 机时之间的权衡 —— 但视频也会一起没了，所以停实例之前记得把 `outputs/` 从挂载上
拷走。

## 附记：开出这台八卡机器

上面的两组用法假定机器上有八张 A100。在 NVIDIA Brev 上，这种机型来自 academic-grant
预留池，而这个池子对 `brev search` 是不可见的 —— search 读的是公共 catalog，而
grant 专属机型绑定在组织上，只出现在该组织自己的可用机型列表里。所以别去搜，直接点名：

```bash
brev set "<grant org>"

# 确认机型能被接受。什么都不会启动，也不消耗 grant 机时。
brev create sana-8xa100 --type azurerm.a100x8.sxm.academic-grant --dry-run

brev create sana-8xa100 --type azurerm.a100x8.sxm.academic-grant \
  --startup-script @provision.sh

brev refresh          # 只有在网页控制台里创建的实例才需要
ssh sana-8xa100
```

`ssh <name>` 之所以能解析，是因为 CLI 维护着 `~/.brev/ssh_config` —— 每个实例一个 host
条目、一个 cloudflared `ProxyCommand`、以及 `IdentityFile ~/.brev/brev.pem` —— 并且在
`~/.ssh/config` 里加了指向该文件的 `Include`。没有任何东西会自己去读 `~/.brev/`。
`ssh -G sana-8xa100` 会在不连接的情况下打印解析后的配置，这是区分 `Include` 顺序问题和
实例已死的办法。

有四件事决定上面那次批量运行在拿到的机器上能不能跑起来：

- 默认的 SSH 目标是实例的容器，不是那台 GPU 虚拟机。批量运行需要的是宿主：八张可见的卡、
  固定的 SGLang 镜像、以及那块大盘。用 `brev shell sana-8xa100 --host`，并且在相信任何
  `H3_GPU_GROUPS` 取值之前先在那里读一遍 `nvidia-smi -L`。
- Software Configuration 面板问你的那两个版本号，都不是模型实际运行的版本。在
  **Single Container** 下，Brev Container 的 Python（3.10）和 CUDA（12.0.1）描述的是那个
  容器；运行发生在固定的 SGLang 镜像内，它自带 CUDA 13、torch `2.11.0+cu130` 和
  Triton `3.6.0`。建议直接选 **VM Mode**，让 shell 落在宿主上。宿主真正必须提供的，是一个
  新到足以跑 cu13 镜像的驱动 —— 直接执行 `nvidia-smi`，表头那个 `CUDA Version:` 必须是
  13.x，镜像里没有任何东西能让偏旧的驱动跑起来。Python 那边，第 2 步的
  `uv venv --python 3.12` 会自己取解释器，所以面板给的 3.10 无所谓；唯一要紧的是 launcher
  最终跑在 3.11 或更新的版本上，因为 manifest 形态的配置需要 `tomllib`（或 `tomli` 后向兼容包）。
- 在这个机型上，`[0,1,2,3], [4,5,6,7]` 同时也是 CPU socket 的边界，所以它就是第 5 步的
  `nvidia-smi topo -m` 应该印证的那个划分。96 个 vCPU 也正好能被两组各自已经申请的
  `cpus_per_task = 48` 分完。
- Grant 机时从实例开始运行的那一刻就开始计费，所以镜像拉取和 269 GiB 的 checkpoint 下载
  应该放进 `--startup-script`，而不是放在交互会话里做。`H3_STORAGE_ROOT` 必须指向那块大盘，
  因为 prompt 列表、`frames/` 和输出目录都得在它下面，容器模式才解析得到。
