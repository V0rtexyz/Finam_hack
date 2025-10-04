from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Tuple

from app.config import cfg
from app.data_loader import get_raw_message_text
from app.investor_focus import generate_investor_focus, make_text_key


def _resolve_dataset_paths(dataset_dir: Path, names: list[str]) -> list[Path]:
    if not names:
        return sorted(dataset_dir.glob("*.json"))

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_name in names:
        if not raw_name:
            continue
        candidates: list[Path] = []
        candidate = dataset_dir / raw_name
        if candidate.exists():
            candidates.append(candidate)
        candidate_with_ext = dataset_dir / f"{raw_name}.json"
        if candidate_with_ext.exists():
            candidates.append(candidate_with_ext)
        candidates.extend(sorted(dataset_dir.glob(raw_name)))
        if not candidates:
            raise FileNotFoundError(f"Dataset '{raw_name}' not found in {dataset_dir}")
        for path in candidates:
            if path.suffix.lower() != ".json":
                continue
            if path not in seen:
                resolved.append(path)
                seen.add(path)
    return resolved


def _extract_messages(payload: Any) -> Tuple[Any, list[Any]]:
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Dataset dictionary must contain a 'messages' list")
        return payload, messages
    if isinstance(payload, list):
        return None, payload
    raise ValueError("Unsupported dataset format: expected list or dict")


def _output_path(path: Path, *, in_place: bool, suffix: str) -> Path:
    if in_place:
        return path
    suffix = suffix.strip()
    if suffix:
        filename = f"{path.stem}.{suffix}{path.suffix}"
    else:
        filename = path.name
    return path.with_name(filename)


def _collect_jobs(messages: list[Any]) -> tuple[list[tuple[int, str]], dict[str, str]]:
    jobs: list[tuple[int, str]] = []
    unique_map: dict[str, str] = {}

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        text = get_raw_message_text(message).strip()
        if not text:
            continue
        text_hash = make_text_key(text)
        existing_hash = str(message.get("investor_focus_hash") or "")
        if message.get("investor_focus") and existing_hash == text_hash:
            continue
        jobs.append((idx, text_hash))
        if text_hash not in unique_map:
            unique_map[text_hash] = text
    return jobs, unique_map


def _apply_focus(messages: list[Any], jobs: list[tuple[int, str]], focus_map: dict[str, str]) -> int:
    updated = 0
    for idx, text_hash in jobs:
        summary = focus_map.get(text_hash, "").strip()
        if not summary:
            continue
        message = messages[idx]
        if isinstance(message, dict):
            message["investor_focus"] = summary
            message["investor_focus_hash"] = text_hash
            updated += 1
    return updated


def process_dataset(path: Path, *, workers: int, in_place: bool, suffix: str) -> tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    container, messages = _extract_messages(payload)

    jobs, unique_map = _collect_jobs(messages)
    if not jobs:
        return 0, 0

    focus_map = generate_investor_focus(unique_map, max_workers=workers)
    updated = _apply_focus(messages, jobs, focus_map)
    if not updated:
        return 0, len(jobs)

    # Persist changes
    target_path = _output_path(path, in_place=in_place, suffix=suffix)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if container is not None:
        container["messages"] = messages
        serialized = container
    else:
        serialized = messages
    target_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated, len(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich dataset messages with investor-focused LLM summaries.")
    parser.add_argument("datasets", nargs="*", help="Specific dataset filenames or glob patterns (default: all *.json in dataset dir).")
    parser.add_argument("--dataset-dir", type=Path, default=cfg.dataset_dir, help="Directory with dataset JSON files (default: %(default)s)")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel LLM requests (default: auto-detect by CPU count)",
    )
    parser.add_argument("--suffix", type=str, default="focused", help="Suffix for generated files when not using --in-place (default: %(default)s)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite source files instead of creating suffixed copies.")

    args = parser.parse_args()
    auto_workers = max(1, min(16, (os.cpu_count() or 4)))
    worker_count = auto_workers if not args.workers else max(1, args.workers)
    if args.workers and args.workers < 1:
        raise SystemExit("--workers must be a positive integer")
    print(f"[INFO] Using {worker_count} parallel worker(s)")
    dataset_dir: Path = args.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    paths = _resolve_dataset_paths(dataset_dir, args.datasets)
    if not paths:
        print("No dataset files found.")
        return

    total_candidates = 0
    total_updated = 0
    for dataset_path in paths:
        try:
            updated, candidates = process_dataset(
                dataset_path,
                workers=worker_count,
                in_place=args.in_place,
                suffix=args.suffix,
            )
        except Exception as exc:
            print(f"[ERROR] {dataset_path.name}: {exc}")
            continue
        total_candidates += candidates
        total_updated += updated
        if candidates == 0:
            print(f"[SKIP] {dataset_path.name}: nothing to update")
        elif updated == 0:
            print(f"[NO CHANGE] {dataset_path.name}: {candidates} messages already enriched")
        else:
            location = dataset_path if args.in_place else _output_path(dataset_path, in_place=False, suffix=args.suffix)
            print(f"[OK] {dataset_path.name}: {updated}/{candidates} messages enriched -> {location.name}")

    print(f"Done. Updated {total_updated} of {total_candidates} candidate messages across {len(paths)} file(s).")


if __name__ == "__main__":
    main()
