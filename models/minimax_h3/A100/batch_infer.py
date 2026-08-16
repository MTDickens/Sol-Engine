#!/usr/bin/env python3
"""Fan a MiniMax-H3 prompt list out over one or more four-GPU groups.

`gpu_infer.py` runs a single prompt through one warmup and one measured request
because it exists to produce a benchmark number. This launcher is the batch
counterpart: it reads a prompt list, splits it into even contiguous shards -- one
per GPU group -- and runs `batch_worker.py` once per shard with
CUDA_VISIBLE_DEVICES pinned to that group. Each worker loads the 33B checkpoint
once and generates its whole shard, so N prompts cost one model load per group
instead of N.

Nothing here imports torch or SGLang. The parent only shards work and merges
results, which keeps the "exactly four visible GPUs" check inside the workers,
where it is true.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


RUNTIME_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNTIME_ROOT.parents[2]
WORKER = RUNTIME_ROOT / "batch_worker.py"
DEFAULT_PROMPTS_FILE = REPO_ROOT / "models/minimax_h3/demo_prompts.json"

# One group is any parenthesised or bracketed run of device ids, so both
# "(0,1,2,3), (4,5,6,7)" and "[0,1,2,3], [4,5,6,7]" parse, and a bare
# "0,1,2,3" is read as the single group it obviously is.
GROUP_RE = re.compile(r"[\[(]\s*([^\[\]()]*?)\s*[\])]")
# Video names become file names in outputs/, so no separators and no leading dot.
VIDEO_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _fail(message: str) -> SystemExit:
    return SystemExit(f"batch_infer: {message}")


def _video_name(value: Any, position: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"item {position} has no non-empty video_name")
    name = value.strip()
    # Accept both "clip_a" and "clip_a.mp4" for the same output file.
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    if not VIDEO_NAME_RE.fullmatch(name):
        raise _fail(
            f"item {position} has an unusable video_name {value!r}: use letters, "
            "digits, '.', '_' and '-' only, starting with a letter or digit"
        )
    return name


def _prompt(value: Any, position: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"item {position} has no non-empty prompt")
    return value


def _seed(raw: dict[str, Any], position: int) -> int | None:
    if "seed" not in raw or raw["seed"] is None:
        return None
    try:
        return int(raw["seed"])
    except (TypeError, ValueError) as exc:
        raise _fail(f"item {position} has a non-integer seed: {raw['seed']!r}") from exc


def _image(raw: dict[str, Any], position: int) -> str | None:
    value = raw.get("image")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"item {position} has an empty image path")
    return value.strip()


def _entry(raw: Any, position: int) -> dict[str, Any]:
    if isinstance(raw, (list, tuple)):
        # The list dialect is the (prompt, video_name) tuple written out in JSON.
        # A first frame needs the object form; two fields have nowhere to put it.
        if len(raw) != 2:
            raise _fail(
                f"item {position} must be [prompt, video_name]; it has {len(raw)} fields"
            )
        return {
            "video_name": _video_name(raw[1], position),
            "prompt": _prompt(raw[0], position),
            "seed": None,
            "image": None,
        }
    if isinstance(raw, dict):
        return {
            "video_name": _video_name(raw.get("video_name"), position),
            "prompt": _prompt(raw.get("prompt"), position),
            "seed": _seed(raw, position),
            "image": _image(raw, position),
        }
    raise _fail(f"item {position} must be an object or a [prompt, video_name] pair")


def load_items(path: Path) -> list[dict[str, Any]]:
    """Read the prompt list. Accepts a bare list or {"items": [...]}."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _fail(f"prompt list does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise _fail(f"prompt list is not valid JSON: {path}: {exc}") from exc

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("items")
        if entries is None:
            raise _fail(f"prompt list has no 'items' array: {path}")
    else:
        raise _fail(f"prompt list must be a list or an object: {path}")
    if not isinstance(entries, list) or not entries:
        raise _fail(f"prompt list is empty: {path}")

    items = []
    seen: dict[str, int] = {}
    for position, raw in enumerate(entries):
        item = _entry(raw, position)
        name = item["video_name"]
        if name in seen:
            raise _fail(
                f"video_name {name!r} is used by items {seen[name]} and {position}; "
                "they would overwrite the same mp4"
            )
        seen[name] = position
        item["index"] = position
        if item["image"] is not None:
            # Relative to the prompt list, which is the one path the launcher
            # already holds in whatever namespace it is running in. Container
            # runs see the list under /h3, so a relative first frame resolves
            # there too and no host path has to be rewritten into the JSON.
            image = Path(item["image"])
            if not image.is_absolute():
                image = path.parent / image
            image = image.resolve()
            if not image.is_file():
                raise _fail(
                    f"item {position} ({name}) points at a first frame that does not "
                    f"exist: {image}"
                )
            item["image"] = str(image)
        items.append(item)
    return items


def parse_gpu_groups(raw: str, group_size: int) -> list[list[int]]:
    """Parse H3_GPU_GROUPS into disjoint groups of exactly `group_size` ids."""

    text = (raw or "").strip()
    if not text:
        raise _fail("H3_GPU_GROUPS is empty")
    chunks = GROUP_RE.findall(text)
    if chunks:
        leftover = GROUP_RE.sub("", text).strip(" \t,;")
        if leftover:
            raise _fail(f"H3_GPU_GROUPS has text outside its groups: {raw!r}")
    else:
        chunks = [text]

    groups: list[list[int]] = []
    owner: dict[int, int] = {}
    for position, chunk in enumerate(chunks):
        tokens = [token.strip() for token in chunk.split(",") if token.strip()]
        if not tokens:
            raise _fail(f"group {position} in H3_GPU_GROUPS is empty")
        group: list[int] = []
        for token in tokens:
            if not token.isdigit():
                raise _fail(f"group {position} has a non-numeric GPU id: {token!r}")
            device = int(token)
            if device in owner:
                raise _fail(
                    f"GPU {device} appears in group {owner[device]} and group "
                    f"{position}; groups must be disjoint"
                )
            owner[device] = position
            group.append(device)
        if len(group) != group_size:
            raise _fail(
                f"group {position} has {len(group)} GPUs but the config pins "
                f"{group_size} per run ({group}); every group must be that wide"
            )
        groups.append(group)
    return groups


def resolve_devices(groups: list[list[int]]) -> list[list[str]]:
    """Map group ids through an inherited CUDA_VISIBLE_DEVICES, if there is one.

    Under Slurm the job already sees a slice of the node, so a group id is an
    index into that slice rather than a physical device number.
    """

    inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not inherited:
        return [[str(device) for device in group] for group in groups]
    visible = [token.strip() for token in inherited.split(",") if token.strip()]
    resolved = []
    for group in groups:
        row = []
        for device in group:
            if device >= len(visible):
                raise _fail(
                    f"GPU {device} is outside the {len(visible)} devices this job "
                    f"was given (CUDA_VISIBLE_DEVICES={inherited})"
                )
            row.append(visible[device])
        resolved.append(row)
    return resolved


def split_even(items: list[Any], count: int) -> list[list[Any]]:
    """Contiguous even split: no load balancing, every item costs the same."""

    size, extra = divmod(len(items), count)
    shards = []
    start = 0
    for index in range(count):
        stop = start + size + (1 if index < extra else 0)
        shards.append(items[start:stop])
        start = stop
    return shards


def _stream(pipe: Any, prefix: str, log_path: Path, lock: threading.Lock) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        for line in pipe:
            handle.write(line)
            handle.flush()
            with lock:
                sys.stdout.write(prefix + line)
                sys.stdout.flush()


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", str(RUNTIME_ROOT / "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_file = Path(os.environ.get("H3_PROMPTS_FILE", str(DEFAULT_PROMPTS_FILE)))
    items = load_items(prompts_file)

    group_size = int(os.environ.get("H3_GPUS_PER_GROUP", "4"))
    groups = parse_gpu_groups(os.environ.get("H3_GPU_GROUPS", "[0,1,2,3]"), group_size)
    devices = resolve_devices(groups)
    shards = split_even(items, len(groups))

    print(
        f"batch_infer: {len(items)} prompts from {prompts_file} across "
        f"{len(groups)} group(s) of {group_size} GPUs",
        flush=True,
    )
    for index, (group, shard) in enumerate(zip(groups, shards)):
        names = ", ".join(item["video_name"] for item in shard) or "(nothing to do)"
        print(f"  group {index} gpus={group}: {len(shard)} item(s) -> {names}", flush=True)

    shard_dir = out_dir / "batch_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    python_bin = os.environ.get("H3_PYTHON_BIN", "python3")
    base_port = int(os.environ.get("H3_MASTER_PORT", "30005"))
    lock = threading.Lock()
    started = time.monotonic()
    running = []
    for index, (group, device_row, shard) in enumerate(zip(groups, devices, shards)):
        if not shard:
            # More groups than prompts. Spawning here would load the checkpoint
            # onto four GPUs just to generate nothing.
            continue
        shard_path = shard_dir / f"group{index}_shard.json"
        result_path = shard_dir / f"group{index}_result.json"
        log_path = out_dir / f"group{index}.log"
        shard_path.write_text(
            json.dumps(
                {
                    "group": index,
                    "gpus": group,
                    "cuda_visible_devices": ",".join(device_row),
                    "items": shard,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(device_row)
        env["H3_BATCH_GROUP_INDEX"] = str(index)
        env["H3_BATCH_SHARD_FILE"] = str(shard_path)
        env["H3_BATCH_RESULT_FILE"] = str(result_path)
        # Groups are independent torch.distributed jobs on one node, so each one
        # needs its own rendezvous port.
        env["H3_MASTER_PORT"] = str(base_port + index * 100)
        process = subprocess.Popen(
            [python_bin, str(WORKER)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        reader = threading.Thread(
            target=_stream,
            args=(process.stdout, f"[g{index}] ", log_path, lock),
            daemon=True,
        )
        reader.start()
        running.append(
            {
                "index": index,
                "gpus": group,
                "cuda_visible_devices": ",".join(device_row),
                "process": process,
                "reader": reader,
                "result_path": result_path,
                "log": str(log_path),
                "shard": shard,
            }
        )

    for entry in running:
        entry["returncode"] = entry["process"].wait()
        entry["reader"].join()
    wall_s = time.monotonic() - started

    group_records = []
    item_records = []
    shared: dict[str, Any] = {}
    for entry in running:
        result = None
        if entry["result_path"].is_file():
            try:
                result = json.loads(entry["result_path"].read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result = None
        record = {
            "index": entry["index"],
            "gpus": entry["gpus"],
            "cuda_visible_devices": entry["cuda_visible_devices"],
            "returncode": entry["returncode"],
            "log": entry["log"],
            "item_count": len(entry["shard"]),
        }
        if result is None:
            record["status"] = "no_result"
            item_records.extend(
                {
                    "index": item["index"],
                    "video_name": item["video_name"],
                    "group": entry["index"],
                    "status": "no_result",
                }
                for item in entry["shard"]
            )
        else:
            record["status"] = result.get("status", "unknown")
            record["wall_s"] = result.get("wall_s")
            record["warmup"] = result.get("warmup")
            record["route_density"] = result.get("route_density")
            for item in result.get("items", []):
                item_records.append({**item, "group": entry["index"]})
            for key in ("profile", "hardware", "runtime", "workload"):
                shared.setdefault(key, result.get(key))
        group_records.append(record)

    item_records.sort(key=lambda record: record["index"])
    ok_items = [record for record in item_records if record.get("status") == "ok"]
    batch = {
        "schema_version": 1,
        **shared,
        "prompts_file": str(prompts_file),
        "prompt_count": len(items),
        "gpus_per_group": group_size,
        "gpu_groups": groups,
        "wall_s": wall_s,
        "generated": len(ok_items),
        "groups": group_records,
        "items": item_records,
    }
    batch_path = out_dir / "batch.json"
    batch_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failures = [record for record in item_records if record.get("status") != "ok"]
    print(
        f"\nbatch_infer: {len(ok_items)}/{len(items)} videos in {wall_s:.1f}s -> {out_dir}",
        flush=True,
    )
    for record in item_records:
        mark = "ok " if record.get("status") == "ok" else "FAIL"
        seconds = record.get("inference_time_s")
        timing = f"{seconds:8.2f}s" if isinstance(seconds, (int, float)) else " " * 9
        note = "" if record.get("status") == "ok" else f"  {record.get('error', record.get('status'))}"
        print(f"  {mark} g{record.get('group')} {timing}  {record['video_name']}.mp4{note}", flush=True)
    print(f"batch_infer: wrote {batch_path}", flush=True)

    unsound = [
        record
        for record in ok_items
        if record.get("sparse_ranks_ok") is False
    ]
    if unsound:
        names = ", ".join(record["video_name"] for record in unsound)
        print(
            "batch_infer: sparse attention did not run on every rank for: "
            f"{names}; the videos are on disk but the profile did not apply cleanly",
            file=sys.stderr,
            flush=True,
        )
    if failures or unsound or any(entry["returncode"] != 0 for entry in running):
        return 1
    shutil.rmtree(shard_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
