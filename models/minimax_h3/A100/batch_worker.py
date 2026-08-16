#!/usr/bin/env python3
"""Generate one shard of a MiniMax-H3 prompt list on four GPUs.

Started by `batch_infer.py`, one process per GPU group, with
CUDA_VISIBLE_DEVICES already pinned to that group's four devices. The model is
loaded once and every prompt in the shard is generated through the same
DiffGenerator, which is the whole point of batch mode.

Registration stays at module scope for the same reason it does in
`gpu_infer.py`: SGLang launches its workers with multiprocessing spawn and each
one re-imports this module. Importing gpu_infer is what performs it, and reusing
that module's sampling parameters is deliberate -- a batch video comes out of
the same 768p 16:9 target, the same flow shifts and the same step count as the
benchmarked one, so the two paths cannot drift apart.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


RUNTIME_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNTIME_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.minimax_h3.A100 import gpu_infer  # noqa: E402


HARDWARE = gpu_infer.HARDWARE
PROFILE = gpu_infer.PROFILE
TRUE_VALUES = {"1", "true", "yes", "on"}
IMAGE_PLACEHOLDER = "{image}"
DEFAULT_IMAGE_CONDITION = {"type": "image", "role": "first_frame", "path": IMAGE_PLACEHOLDER}


def _substitute(value: Any, image: str) -> Any:
    if isinstance(value, str):
        return value.replace(IMAGE_PLACEHOLDER, image)
    if isinstance(value, list):
        return [_substitute(element, image) for element in value]
    if isinstance(value, dict):
        return {key: _substitute(element, image) for key, element in value.items()}
    return value


def _image_request(image: str) -> tuple[str, list[Any]]:
    """Return the (task, conditions) an item with a first frame is sent with.

    This is the one piece of the batch path that could not be checked against
    anything on disk: no runtime in this tree issues a conditioned request, so
    the condition entry's key names come from SGLang's sampling params inside
    the pinned image rather than from a working example here. Both halves are
    therefore environment-overridable, so correcting them is a config change:

        H3_FIRST_FRAME_TASK    task name for a first-frame request (default fl2va,
                               the variant these weights are loaded as)
        H3_IMAGE_CONDITION_JSON  the condition entry as JSON; every "{image}"
                               inside it is replaced with the resolved path

    Whatever is used ends up in batch.json, so a rejected request shows exactly
    what was sent.
    """

    task = os.environ.get("H3_FIRST_FRAME_TASK", "").strip() or "fl2va"
    raw = os.environ.get("H3_IMAGE_CONDITION_JSON", "").strip()
    if raw:
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"H3_IMAGE_CONDITION_JSON is not valid JSON: {exc}") from exc
    else:
        entry = DEFAULT_IMAGE_CONDITION
    conditions = entry if isinstance(entry, list) else [entry]
    return task, _substitute(conditions, image)


def _event_files(output_dir: Path, group: int) -> list[Path]:
    return sorted(output_dir.glob(f"sol_events_g{group}_rank*.jsonl"))


def _generate(
    generator: Any,
    *,
    epoch_file: Path,
    epoch: str,
    **sampling_params: Any,
) -> dict[str, Any]:
    # The adapter splits requests on this file changing, so every item needs its
    # own epoch string before generate() is called.
    epoch_file.write_text(epoch + "\n", encoding="utf-8")
    result = generator.generate(sampling_params_kwargs=sampling_params)
    if result is None or isinstance(result, list):
        raise RuntimeError(f"Expected one MiniMax-H3 result, got {result!r}")
    return gpu_infer._result_record(result)


def _collect_events(
    output_dir: Path,
    group: int,
    records: list[dict[str, Any]],
    warmup_epoch: str | None,
) -> dict[str, Any] | None:
    """Attach per-item sparse/cache telemetry from this group's event logs."""

    if not PROFILE.sol_attention:
        return None
    events: list[dict[str, Any]] = []
    for path in _event_files(output_dir, group):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))

    expected_ranks = set(range(int(os.environ["H3_NUM_GPUS"])))
    sparse_ranks: dict[str, set[int]] = {}
    for event in events:
        if event.get("event") == "first_sparse_forward":
            key = str(event.get("request_epoch", ""))
            sparse_ranks.setdefault(key, set()).add(int(event["rank"]))

    decision_event = None
    cache_kind = None
    if PROFILE.cache == "easycache":
        cache_kind, decision_event = "easycache", "easycache_decision"
    elif PROFILE.cache == "firstblock":
        cache_kind, decision_event = "firstblockcache", "firstblockcache_decision"
    decisions: dict[str, list[dict[str, Any]]] = {}
    if decision_event is not None:
        for event in events:
            if event.get("event") == decision_event and int(event.get("rank", -1)) == 0:
                decisions.setdefault(str(event.get("request_epoch", "")), []).append(event)

    for record in records:
        if record["status"] != "ok":
            continue
        epoch = record["request_epoch"]
        ranks = sorted(sparse_ranks.get(epoch, set()))
        record["sparse_ranks"] = ranks
        # Reported rather than raised: unlike the benchmark path, the mp4 on disk
        # is the deliverable here, so the run finishes and batch_infer exits
        # non-zero with the item named.
        record["sparse_ranks_ok"] = set(ranks) == expected_ranks
        if decision_event is None:
            continue
        calls = decisions.get(epoch, [])
        reuse = sum(event.get("action") == "reuse" for event in calls)
        record["cache"] = {
            "kind": cache_kind,
            "calls": len(calls),
            "compute": len(calls) - reuse,
            "reuse": reuse,
            "reuse_rate": reuse / len(calls) if calls else 0.0,
        }

    density = None
    if warmup_epoch is not None:
        # H3_SOL_DENSITY_MODE is locked to "warmup", so this only exists when a
        # warmup request ran. It is observability: the kernel routes on tau and
        # the threshold type either way.
        density_events = [
            event
            for event in events
            if event.get("event") == "route_density"
            and str(event.get("request_epoch", "")) == warmup_epoch
        ]
        if density_events:
            weights = [int(event["blocks"]) * int(event["heads"]) for event in density_events]
            total_weight = sum(weights)
            density = {
                "scope": "first sparse attention call of this group's warmup request",
                "threshold_density": sum(
                    float(event["threshold_density"]) * weight
                    for event, weight in zip(density_events, weights)
                )
                / total_weight,
                "effective_density": sum(
                    float(event["effective_density"]) * weight
                    for event, weight in zip(density_events, weights)
                )
                / total_weight,
                "sequence_tokens": int(density_events[0]["sequence_tokens"]),
                "sink_tokens": int(density_events[0]["sink_tokens"]),
            }
    return density


def main() -> int:
    group = int(os.environ["H3_BATCH_GROUP_INDEX"])
    shard = json.loads(Path(os.environ["H3_BATCH_SHARD_FILE"]).read_text(encoding="utf-8"))
    result_path = Path(os.environ["H3_BATCH_RESULT_FILE"])
    output_dir = Path(os.environ.get("OUT_DIR", str(RUNTIME_ROOT / "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)

    items = shard["items"]
    steps = int(os.environ.get("H3_MEASURED_NUM_STEPS", "50"))
    warmup_steps = int(os.environ.get("H3_WARMUP_NUM_STEPS", str(steps)))
    duration_s = float(os.environ.get("H3_DURATION_SECONDS", "5.166667"))
    default_seed = int(os.environ.get("H3_SEED", "0"))
    warmup_seed = int(os.environ.get("H3_WARMUP_SEED", str(default_seed + 10_000)))
    # Off by default: a warmup is a full generation, which doubles the cost of a
    # short batch and buys nothing that ends up on disk. Turn it on to restore
    # the benchmark path's steady-state timings and route-density telemetry.
    warmup_enabled = os.environ.get("H3_BATCH_WARMUP", "0").strip().lower() in TRUE_VALUES
    os.environ.setdefault("H3_EASYCACHE_NUM_FORWARDS", str(steps - 1))

    # Both files are per group: the epoch file is how the adapter detects a new
    # request and the event log is keyed only by rank, so two groups sharing an
    # output directory would otherwise overwrite each other's state.
    epoch_file = output_dir / f"request_epoch_g{group}.txt"
    os.environ["H3_REQUEST_EPOCH_FILE"] = str(epoch_file)
    os.environ["H3_SOL_EVENT_LOG"] = str(output_dir / ("sol_events_g%d_rank{rank}.jsonl" % group))
    for path in _event_files(output_dir, group):
        path.unlink()

    records: list[dict[str, Any]] = []
    warmup_record: dict[str, Any] | None = None
    warmup_epoch: str | None = None
    density: dict[str, Any] | None = None
    status = "failed"
    started = time.monotonic()
    runtime = gpu_infer._validate_runtime()
    model_path = os.environ.get("H3_MODEL_PATH", "MiniMaxAI/MiniMax-H3")
    model_subfolder = os.environ.get("H3_MODEL_SUBFOLDER")
    if not model_subfolder:
        model_subfolder = "." if Path(model_path).name.lower() == "fl2va" else "FL2VA"
    generator = gpu_infer.DiffGenerator.from_pretrained(
        local_mode=True,
        model_path=model_path,
        model_subfolder=model_subfolder,
        model_variant="fl2va",
        revision=os.environ["H3_MODEL_REVISION"],
        num_gpus=4,
        tp_size=1,
        ulysses_degree=4,
        enable_cfg_parallel=False,
        performance_mode="speed",
        use_fsdp_inference=True,
        layerwise_offload_components=[],
        enable_torch_compile=False,
        regional_compile=False,
        server_warmup=False,
        master_port=int(os.environ.get("H3_MASTER_PORT", "30005")),
    )
    try:
        if warmup_enabled and items:
            warmup_epoch = f"{PROFILE.name}:warmup:{group}:{time.time_ns()}"
            warmup_name = f"warmup_g{group}.mp4"
            print(f"warmup on {items[0]['video_name']} ({warmup_steps} steps)", flush=True)
            warmup_record = _generate(
                generator,
                epoch_file=epoch_file,
                epoch=warmup_epoch,
                **gpu_infer._sampling_params(
                    prompt=items[0]["prompt"],
                    output_dir=output_dir,
                    output_name=warmup_name,
                    steps=warmup_steps,
                    duration_s=duration_s,
                    seed=warmup_seed,
                    save_output=True,
                ),
            )
            warmup_record["output_file"] = None
            (output_dir / warmup_name).unlink(missing_ok=True)

        for position, item in enumerate(items):
            name = item["video_name"]
            seed = default_seed if item.get("seed") is None else int(item["seed"])
            epoch = f"{PROFILE.name}:measured:{group}:{item['index']:04d}:{time.time_ns()}"
            image = item.get("image")
            record: dict[str, Any] = {
                "index": item["index"],
                "video_name": name,
                "seed": seed,
                "prompt_sha256": hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest(),
                "request_epoch": epoch,
                "image": image,
                "status": "ok",
            }
            # Everything except the conditioning comes from gpu_infer, so a
            # first-frame item is the same 768p 16:9 generation as a text-only one.
            params = gpu_infer._sampling_params(
                prompt=item["prompt"],
                output_dir=output_dir,
                output_name=f"{name}.mp4",
                steps=steps,
                duration_s=duration_s,
                seed=seed,
                save_output=True,
            )
            if image is not None:
                params["task"], params["conditions"] = _image_request(image)
            record["task"] = params["task"]
            record["conditions"] = params["conditions"]
            print(
                f"[{position + 1}/{len(items)}] {name}.mp4  seed={seed} steps={steps} "
                f"task={params['task']}" + (f" first_frame={Path(image).name}" if image else ""),
                flush=True,
            )
            try:
                record.update(
                    _generate(
                        generator,
                        epoch_file=epoch_file,
                        epoch=epoch,
                        **params,
                    )
                )
                record["video_exists"] = (output_dir / f"{name}.mp4").is_file()
                print(
                    f"[{position + 1}/{len(items)}] {name}.mp4 done in "
                    f"{record['inference_time_s']:.2f}s",
                    flush=True,
                )
            except Exception as exc:  # one bad prompt must not discard the rest
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            records.append(record)
        status = "ok" if all(record["status"] == "ok" for record in records) else "partial"
    finally:
        generator.shutdown()
        try:
            density = _collect_events(output_dir, group, records, warmup_epoch)
        except Exception as exc:  # telemetry must never sink finished videos
            print(f"event collection failed: {exc}", file=sys.stderr, flush=True)
        for path in _event_files(output_dir, group):
            path.unlink()
        epoch_file.unlink(missing_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "group": group,
                    "gpus": shard["gpus"],
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "status": status,
                    "wall_s": time.monotonic() - started,
                    "profile": asdict(PROFILE),
                    "hardware": asdict(HARDWARE),
                    "runtime": runtime,
                    "model": {
                        "repo_or_path": model_path,
                        "subfolder": model_subfolder,
                        "revision": os.environ["H3_MODEL_REVISION"],
                        "partition": "FL2VA",
                        "dtype": "bfloat16",
                    },
                    "workload": {
                        "task": "t2va",
                        "width": 1344,
                        "height": 768,
                        "frames": 124,
                        "duration_s": duration_s,
                        "measured_steps": steps,
                    },
                    "warmup": warmup_record,
                    "route_density": density,
                    "items": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
