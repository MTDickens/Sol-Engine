#!/usr/bin/env python3
"""Turn the StEvo-Bench task tree into a MiniMax-H3 batch prompt list.

StEvo-Bench ships one directory per task, holding a YAML spec and the initial
frame the video is supposed to evolve from:

    image_implied/alka_seltzer_fizz_cardboard/
        alka_seltzer_fizz_cardboard.yaml
        alka_seltzer_fizz_cardboard_init_frame.png

This script downloads the dataset, reads `id` and `prompts.video_WM` out of each
YAML, copies the initial frame next to the prompt list it writes, and deletes the
download again. The result is exactly what run_minimax_h3_batch.sh consumes:

    <out-dir>/prompts.json      {"items": [{video_name, prompt, image}, ...]}
    <out-dir>/frames/<id>.png

Image paths in the list are relative to the list itself, which is what makes the
output portable into the container: the batch launcher resolves them under /h3.

    python3 scripts/build_stevo_bench_prompts.py
    python3 scripts/build_stevo_bench_prompts.py --tasks image_implied
    python3 scripts/build_stevo_bench_prompts.py --tasks   # neither

The download is a snapshot, not a clone, so no .git comes with it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SETS = ("image_implied", "simple_trigger")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
# The same rule the batch launcher enforces, applied here so a bad id is caught
# while the fix is still cheap.
VIDEO_NAME_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def log(message: str) -> None:
    print(message, flush=True)


def find_task_set(snapshot: Path, name: str) -> Path | None:
    """Locate a task set. The HF tree keeps them at the root, a checkout of the
    benchmark keeps them under benchmark/tasks/, so look for either."""

    direct = snapshot / name
    if direct.is_dir():
        return direct
    for candidate in sorted(snapshot.rglob(name)):
        if candidate.is_dir():
            return candidate
    return None


def read_task(task_dir: Path) -> dict[str, object] | tuple[None, str]:
    import yaml

    specs = sorted(task_dir.glob("*.yaml")) + sorted(task_dir.glob("*.yml"))
    if not specs:
        return None, "no yaml"
    spec = yaml.safe_load(specs[0].read_text(encoding="utf-8")) or {}

    task_id = str(spec.get("id") or task_dir.name).strip()
    if not task_id or set(task_id) - VIDEO_NAME_OK or not task_id[0].isalnum():
        return None, f"unusable id {task_id!r}"

    prompt = (spec.get("prompts") or {}).get("video_WM")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "empty prompts.video_WM"

    images = [
        path
        for path in sorted(task_dir.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
    ]
    if not images:
        return None, "no initial frame"
    # A task directory holds one frame; prefer the one that says so if it grows.
    frame = next((path for path in images if "init" in path.name.lower()), images[0])

    return {
        "id": task_id,
        "prompt": " ".join(prompt.split()),
        "frame": frame,
        "level": spec.get("level"),
        "category": spec.get("category"),
    }, ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=TASK_SETS,
        default=list(TASK_SETS),
        help="task sets to convert; both by default, and `--tasks` with no value converts neither",
    )
    parser.add_argument(
        "--out-dir",
        default="models/minimax_h3/stevo_bench",
        help="where prompts.json and frames/ are written (repo-relative unless absolute)",
    )
    parser.add_argument("--repo-id", default="JhanLiufu/StEvo-Bench")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--keep-snapshot",
        metavar="DIR",
        help="download here and leave it in place; by default the download is a "
        "temporary directory that is deleted once the frames are copied out",
    )
    args = parser.parse_args()

    selected = list(dict.fromkeys(args.tasks))
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    frames_dir = out_dir / "frames"
    prompts_path = out_dir / "prompts.json"

    # Regenerated wholesale: leaving frames from a previous, wider selection
    # behind would quietly contradict the list next to them.
    if frames_dir.exists():
        log(f"replacing {frames_dir}")
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, object]] = []
    skipped: list[str] = []
    if not selected:
        log("no task set selected: writing an empty prompt list, downloading nothing")
    else:
        from huggingface_hub import snapshot_download

        download_dir = Path(args.keep_snapshot).expanduser() if args.keep_snapshot else None
        temp_dir = None if download_dir else tempfile.mkdtemp(prefix="stevo-bench-")
        snapshot = download_dir or Path(temp_dir)
        try:
            log(f"downloading {args.repo_id}@{args.revision} -> {snapshot}")
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                local_dir=str(snapshot),
                allow_patterns=[f"{name}/**" for name in selected]
                + [f"**/{name}/**" for name in selected],
            )

            seen: dict[str, str] = {}
            for name in selected:
                task_set = find_task_set(snapshot, name)
                if task_set is None:
                    log(f"  {name}: not found in the snapshot")
                    continue
                task_dirs = sorted(path for path in task_set.iterdir() if path.is_dir())
                log(f"  {name}: {len(task_dirs)} task(s)")
                for task_dir in task_dirs:
                    task, reason = read_task(task_dir)
                    if task is None:
                        skipped.append(f"{name}/{task_dir.name}: {reason}")
                        continue
                    task_id = str(task["id"])
                    if task_id in seen:
                        # video_name is the output file name, so a collision
                        # would have one task overwrite the other's mp4.
                        skipped.append(
                            f"{name}/{task_dir.name}: id {task_id!r} already taken by {seen[task_id]}"
                        )
                        continue
                    seen[task_id] = f"{name}/{task_dir.name}"
                    frame = task["frame"]
                    destination = frames_dir / f"{task_id}{frame.suffix.lower()}"
                    shutil.copyfile(frame, destination)
                    items.append(
                        {
                            "video_name": task_id,
                            "prompt": task["prompt"],
                            "image": f"frames/{destination.name}",
                            "task_set": name,
                            "level": task["level"],
                            "category": task["category"],
                        }
                    )
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
                log(f"removed {temp_dir}")

    items.sort(key=lambda item: (item["task_set"], item["video_name"]))
    prompts_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": (
                    f"StEvo-Bench {'+'.join(selected) if selected else 'nothing'} as a "
                    "MiniMax-H3 batch prompt list. Generated by "
                    "scripts/build_stevo_bench_prompts.py; prompt is the "
                    "task's prompts.video_WM and image is its initial frame. "
                    "task_set/level/category are provenance and are ignored by the runner."
                ),
                "source": {"repo_id": args.repo_id, "revision": args.revision},
                "items": items,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    log(f"\nwrote {prompts_path}: {len(items)} item(s)")
    log(f"wrote {frames_dir}: {len(list(frames_dir.iterdir()))} frame(s)")
    if skipped:
        log(f"skipped {len(skipped)}:")
        for line in skipped:
            log(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
