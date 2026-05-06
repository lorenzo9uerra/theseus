#!/usr/bin/env python3
"""
Package minimal MAGIC artifacts for eval-only reproducibility.

This bundles the processed per-dataset DGL data directories, paper checkpoints,
and optional evaluation logs.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

DATASETS = ("cadets", "fivedirections", "theia", "trace")
SEEDS = (71, 83, 232, 441, 915)
FIXED_EPOCH = 1704067200  # 2024-01-01 00:00:00 UTC
FIXED_ZIP_DATETIME = (2024, 1, 1, 0, 0, 0)
TIMESTAMP_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s*-\s*",
    re.MULTILINE,
)
ABS_PATH_RE = re.compile(r"(?<![\w$])/(?:[^\s'\"`]+)")


def _magic_root() -> Path:
    return Path(__file__).resolve().parent


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _detect_archive_format(out_path: Path) -> str:
    name = out_path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "tar.gz"
    raise ValueError(
        f"Unsupported archive format for {out_path}. Use .zip or .tar.gz/.tgz"
    )


def _zip_add_path(zf: zipfile.ZipFile, src: Path, arc_prefix: str) -> None:
    if src.is_dir():
        for path in sorted(src.rglob("*")):
            if path.is_file():
                rel = path.relative_to(src)
                arcname = f"{arc_prefix}/{rel.as_posix()}"
                info = zipfile.ZipInfo(filename=arcname, date_time=FIXED_ZIP_DATETIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (path.stat().st_mode & 0o777) << 16
                with path.open("rb") as in_fh, zf.open(info, mode="w") as out_fh:
                    shutil.copyfileobj(in_fh, out_fh, length=1024 * 1024)
    else:
        info = zipfile.ZipInfo(filename=arc_prefix, date_time=FIXED_ZIP_DATETIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (src.stat().st_mode & 0o777) << 16
        with src.open("rb") as in_fh, zf.open(info, mode="w") as out_fh:
            shutil.copyfileobj(in_fh, out_fh, length=1024 * 1024)


def _tar_add_path(tf: tarfile.TarFile, src: Path, arc_prefix: str) -> None:
    if src.is_dir():
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            arcname = f"{arc_prefix}/{rel.as_posix()}"
            info = tf.gettarinfo(str(path), arcname=arcname)
            info.mtime = FIXED_EPOCH
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            with path.open("rb") as fh:
                tf.addfile(info, fh)
    else:
        info = tf.gettarinfo(str(src), arcname=arc_prefix)
        info.mtime = FIXED_EPOCH
        info.uid = 0
        info.gid = 0
        info.uname = "root"
        info.gname = "root"
        with src.open("rb") as fh:
            tf.addfile(info, fh)


def _sanitize_log_text(text: str, *, magic_root: Path) -> str:
    replacements = (
        (str(magic_root), "."),
        (str(Path.home()), "$HOME"),
    )
    for src, dst in replacements:
        if src and src != dst:
            text = text.replace(src, dst)

    text = TIMESTAMP_PREFIX_RE.sub("", text)
    text = ABS_PATH_RE.sub("$ABS_PATH", text)
    return text.rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package minimal MAGIC artifacts for eval-only reproducibility."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_magic_root() / "data",
        help="MAGIC processed data directory (default: baselines/MAGIC/data).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_magic_root() / "checkpoints",
        help="MAGIC checkpoint directory (default: baselines/MAGIC/checkpoints).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_magic_root() / "results",
        help="MAGIC evaluation log directory (default: baselines/MAGIC/results).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_magic_root() / "magic_artifacts.zip",
        help="Output archive (.zip or .tar.gz).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS),
        help=f"Comma-separated datasets (default: {','.join(DATASETS)}).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in SEEDS),
        help=f"Comma-separated seeds (default: {','.join(str(seed) for seed in SEEDS)}).",
    )
    parser.add_argument(
        "--no-results",
        action="store_true",
        help="Do not include existing evaluation logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be packaged without writing the archive.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    datasets = _parse_csv_list(args.datasets)
    seeds = [int(seed) for seed in _parse_csv_list(args.seeds)]

    magic_root = _magic_root().resolve()
    to_add: list[tuple[Path, str]] = []
    log_files: dict[str, str] = {}
    manifest_runs: list[dict[str, object]] = []

    for dataset in datasets:
        dataset_dir = data_dir / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Missing required dataset directory: {dataset_dir}"
            )
        to_add.append((dataset_dir, f"data/{dataset}"))

        dataset_logs: list[str] = []
        for seed in seeds:
            checkpoint = checkpoint_dir / f"checkpoint-{dataset}-seed{seed}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Missing required checkpoint file: {checkpoint}"
                )
            to_add.append((checkpoint, f"checkpoints/{checkpoint.name}"))

            if not args.no_results:
                log_path = results_dir / f"{dataset}_seed{seed}.log"
                if log_path.exists():
                    arcname = f"results/{log_path.name}"
                    log_files[arcname] = _sanitize_log_text(
                        log_path.read_text(encoding="utf-8", errors="ignore"),
                        magic_root=magic_root,
                    )
                    dataset_logs.append(arcname)

        manifest_runs.append(
            {
                "dataset": dataset,
                "data_dir": f"data/{dataset}",
                "checkpoints": [
                    f"checkpoints/checkpoint-{dataset}-seed{seed}.pt" for seed in seeds
                ],
                "logs": dataset_logs if not args.no_results else [],
            }
        )

    deduped: list[tuple[Path, str]] = []
    seen_arcs = set()
    for src, arc in to_add:
        if arc in seen_arcs:
            continue
        deduped.append((src, arc))
        seen_arcs.add(arc)

    manifest = {
        "schema": "magic_artifact_bundle_v1",
        "include_results": not args.no_results,
        "runs": manifest_runs,
    }

    if args.dry_run:
        print("Dry run - would package:")
        for src, arc in deduped:
            print(f"  {arc} <= {src}")
        for arc in sorted(log_files):
            print(f"  {arc} <= (sanitized)")
        print(f"Output: {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    archive_format = _detect_archive_format(out_path)
    if archive_format == "zip":
        with zipfile.ZipFile(
            out_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            manifest_info = zipfile.ZipInfo(
                filename="MANIFEST.json", date_time=FIXED_ZIP_DATETIME
            )
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            manifest_info.external_attr = 0o644 << 16
            zf.writestr(manifest_info, json.dumps(manifest, indent=2))
            for arc, content in sorted(log_files.items()):
                info = zipfile.ZipInfo(filename=arc, date_time=FIXED_ZIP_DATETIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, content)
            for src, arc in deduped:
                _zip_add_path(zf, src, arc)
    else:
        with tarfile.open(out_path, mode="w:gz") as tf:
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            info = tarfile.TarInfo("MANIFEST.json")
            info.size = len(manifest_bytes)
            info.mtime = FIXED_EPOCH
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            tf.addfile(info, fileobj=io.BytesIO(manifest_bytes))
            for arc, content in sorted(log_files.items()):
                log_bytes = content.encode("utf-8")
                info = tarfile.TarInfo(arc)
                info.size = len(log_bytes)
                info.mtime = FIXED_EPOCH
                info.uid = 0
                info.gid = 0
                info.uname = "root"
                info.gname = "root"
                tf.addfile(info, fileobj=io.BytesIO(log_bytes))
            for src, arc in deduped:
                _tar_add_path(tf, src, arc)

    print(f"Wrote: {out_path}")
    print(f"Datasets: {len(datasets)} | Entries: {len(deduped)}")


if __name__ == "__main__":
    main()
