#!/usr/bin/env python3
"""Write stable SHA256 manifests for inputs/model/scripts and report outputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--young", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--cc-dir", required=True, type=Path)
    args = p.parse_args()
    audit = args.root / "audit"
    audit.mkdir(exist_ok=True)
    inputs = [
        args.data,
        args.young,
        args.model_dir / "meta.json",
        args.model_dir / "obs_names.npy",
        args.model_dir / "background.npy",
        args.model_dir / "salient.npy",
        args.cc_dir / "meta.json",
        args.cc_dir / "obs_names.npy",
        args.cc_dir / "source_rows.npy",
        args.cc_dir / "spatial.npy",
        args.cc_dir / "salient_spatial.npy",
        *sorted((args.root / "scripts").glob("*.py")),
    ]
    (audit / "input_model_script_sha256.txt").write_text(
        "".join(f"{digest(path)}  {path}\n" for path in inputs)
    )
    excluded = {audit / "output_sha256.txt"}
    outputs = [path for path in sorted(args.root.rglob("*")) if path.is_file() and path not in excluded]
    (audit / "output_sha256.txt").write_text(
        "".join(f"{digest(path)}  {path.relative_to(args.root)}\n" for path in outputs)
    )


if __name__ == "__main__":
    main()
