#!/usr/bin/env python3
"""Clone upstream code into external/ and record the exact installed commits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    repos = json.loads((ROOT / "external/repos.json").read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="*", help="Default: the three paper baseline dependencies.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    names = args.names or list(repos)
    if len(names) != len(set(names)) or any(n not in repos for n in names):
        p.error(f"Choose unique names from {', '.join(repos)}.")
    installed = {}
    for name in names:
        spec = repos[name]
        dest = ROOT / "external" / name
        command = ["git", "clone", "--filter=blob:none", spec["url"], str(dest)]
        if dest.exists():
            if not (dest / ".git").exists():
                raise SystemExit(f"Existing directory is not an upstream clone: {dest}")
            origin = subprocess.check_output(
                ["git", "-C", str(dest), "remote", "get-url", "origin"], text=True).strip()
            revision = subprocess.check_output(
                ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(dest), "status", "--porcelain"], text=True).strip()
            if origin.removesuffix(".git") != spec["url"].removesuffix(".git"):
                raise SystemExit(f"Existing clone has a different origin: {dest}")
            if dirty or (spec["revision"] and revision != spec["revision"]):
                raise SystemExit(f"Existing clone differs from the requested clean revision: {dest}")
            print(f"Using existing {name} at {revision}", flush=True)
            continue
        print(shlex.join(command), flush=True)
        if spec["revision"]:
            print(shlex.join(["git", "-C", str(dest), "checkout", "--detach", spec["revision"]]), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
        if spec["revision"]:
            subprocess.run(["git", "-C", str(dest), "checkout", "--detach", spec["revision"]], check=True)
        revision = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        installed[name] = dict(url=spec["url"], commit=revision,
                               fetched_utc=datetime.now(timezone.utc).isoformat())
        record = ROOT / "external/installed_revisions.json"
        previous = json.loads(record.read_text()) if record.exists() else {}
        record.write_text(json.dumps(previous | installed, indent=2) + "\n")


if __name__ == "__main__":
    main()
