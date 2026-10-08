#!/usr/bin/env python3
"""
tools/import_data.py
====================
Copy run data from the old per-repo data folders into this repo's data/ and
record the basis in every copied config and manifest.

    python tools/import_data.py --from ../2D_eigenbasis_data --basis eigen
    python tools/import_data.py --from ../spinbasis_data_2d  --basis spin
    python tools/import_data.py --from ../2D_diagnostics     --basis diagnostics

Each sweep folder <from>/<sweep> is copied to data/<basis>basis/<sweep>
(diagnostics: <from>/<kind>/<run> -> data/diagnostics/<kind>/<run>). The
SOURCE is only ever read.

Basis backfill, applied to the COPY of every sweep:
  matrix.json          "basis" and runs[i].config.system.basis
  run_*/config.json    system.basis
  run_*/manifest.json  "basis" and config.system.basis
  run_*/*.summary.json "basis"
The sweep's archived config.toml is left verbatim: it records what was
submitted. A config that already names a DIFFERENT basis is an error.

Safe to re-run:
  * a sweep already present in data/ is skipped, unless named with --sweeps,
    in which case it is re-copied from the source and re-backfilled;
  * a LIVE sweep is skipped unless --include-running: one with a run that has
    not finished (no run.done / run.frozen, no final manifest status) and whose
    files changed within --live-hours. That is exactly a sweep still being
    written by a job on the cluster, or one just submitted and queued. Import
    it later, once it has finished:
        python tools/import_data.py --from ... --basis spin --sweeps <sweep> --include-running
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FINAL_STATUSES = {"completed", "failed", "frozen", "crashed_before_setup"}


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    """Rewrite with the indentation every writer in this project uses (2),
    keeping the original's trailing-newline convention."""
    newline = path.read_text().endswith("\n")
    path.write_text(json.dumps(obj, indent=2) + ("\n" if newline else ""))


def set_config_basis(cfg, basis, where):
    system = cfg.setdefault("system", {})
    have = system.get("basis")
    if have not in (None, basis):
        raise SystemExit(f"ERROR: {where} already says basis={have!r}, not {basis!r}")
    system["basis"] = basis


def set_top_basis(obj, basis, where):
    have = obj.get("basis")
    if have not in (None, basis):
        raise SystemExit(f"ERROR: {where} already says basis={have!r}, not {basis!r}")
    obj["basis"] = basis


def newest_mtime(path):
    t = path.stat().st_mtime
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            try:
                t = max(t, os.lstat(os.path.join(root, name)).st_mtime)
            except FileNotFoundError:
                pass
    return t


def run_is_finished(run_dir):
    if (run_dir / "run.done").exists() or (run_dir / "run.frozen").exists():
        return True
    mp = run_dir / "manifest.json"
    if mp.exists():
        try:
            return read_json(mp).get("status") in FINAL_STATUSES
        except Exception:
            return False
    return False


def live_runs(sweep, live_hours):
    """Unfinished runs whose files changed within live_hours."""
    cutoff = time.time() - live_hours * 3600
    out = []
    for rd in sorted(sweep.glob("run_*")):
        if rd.is_dir() and not run_is_finished(rd):
            age = time.time() - newest_mtime(rd)
            if newest_mtime(rd) > cutoff:
                out.append((rd.name, age / 3600))
    if not list(sweep.glob("run_*")) and newest_mtime(sweep) > cutoff:
        out.append(("(no run dirs yet)", (time.time() - newest_mtime(sweep)) / 3600))
    return out


def backfill(sweep, basis):
    """Record the basis everywhere a config or manifest lives. Returns a count."""
    n = 0
    mp = sweep / "matrix.json"
    if mp.exists():
        m = read_json(mp)
        set_top_basis(m, basis, mp)
        for r in m.get("runs", []):
            set_config_basis(r["config"], basis, f"{mp} run {r.get('run_id')}")
        write_json(mp, m); n += 1
    for rd in sorted(sweep.glob("run_*")):
        if not rd.is_dir():
            continue
        cp = rd / "config.json"
        if cp.exists():
            c = read_json(cp); set_config_basis(c, basis, cp); write_json(cp, c); n += 1
        mf = rd / "manifest.json"
        if mf.exists():
            m = read_json(mf)
            set_top_basis(m, basis, mf)
            if isinstance(m.get("config"), dict):
                set_config_basis(m["config"], basis, f"{mf} config")
            write_json(mf, m); n += 1
        for sp in sorted(rd.glob("*.summary.json")):
            s = read_json(sp); set_top_basis(s, basis, sp); write_json(sp, s); n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--from", dest="src", required=True, help="old data root to copy from")
    ap.add_argument("--basis", required=True, choices=["spin", "eigen", "diagnostics"])
    ap.add_argument("--sweeps", nargs="+", default=None,
                    help="only these sweep folder names (re-imports them if already present)")
    ap.add_argument("--include-running", action="store_true",
                    help="also import sweeps that look live (see --live-hours)")
    ap.add_argument("--live-hours", type=float, default=12.0,
                    help="an unfinished run touched within this many hours counts as live (default 12)")
    ap.add_argument("--dry-run", action="store_true", help="report only; copy nothing")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"ERROR: source folder not found: {src}")
    dst_root = REPO / "data" / ("diagnostics" if args.basis == "diagnostics" else f"{args.basis}basis")
    print(f"source:      {src}\ndestination: {dst_root}\n")

    if args.basis == "diagnostics":
        items = [(p.parent.name + "/" + p.name, p) for kind in sorted(src.iterdir()) if kind.is_dir()
                 for p in sorted(kind.iterdir()) if p.is_dir()]
    else:
        items = [(p.name, p) for p in sorted(src.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if args.sweeps:
        wanted = set(args.sweeps)
        items = [(n, p) for n, p in items if n in wanted or p.name in wanted]
        missing = wanted - {n for n, _ in items} - {p.name for _, p in items}
        if missing:
            raise SystemExit(f"ERROR: not found under {src}: {sorted(missing)}")

    tally = {"copied": 0, "re-imported": 0, "skipped (already in data/)": 0, "skipped (live)": 0}
    for name, sweep in items:
        dst = dst_root / name
        if args.basis != "diagnostics":
            live = live_runs(sweep, args.live_hours)
            if live and not args.include_running:
                desc = ", ".join(f"{r} touched {h:.1f} h ago" for r, h in live)
                print(f"  SKIP live      {name}   ({desc}); import later with --include-running")
                tally["skipped (live)"] += 1
                continue
        if dst.exists() and not args.sweeps:
            print(f"  skip present   {name}")
            tally["skipped (already in data/)"] += 1
            continue
        action = "re-import" if dst.exists() else "copy"
        if args.dry_run:
            print(f"  would {action:9s} {name}")
            continue
        # Copy and backfill in a staging folder, so a failure (e.g. a basis
        # conflict) leaves data/ exactly as it was.
        dst.parent.mkdir(parents=True, exist_ok=True)
        stage = dst.parent / f".importing-{dst.name}"
        shutil.rmtree(stage, ignore_errors=True)
        try:
            shutil.copytree(sweep, stage, symlinks=True)
            n = backfill(stage, args.basis) if args.basis != "diagnostics" else 0
            if dst.exists():
                shutil.copytree(stage, dst, symlinks=True, dirs_exist_ok=True)
                shutil.rmtree(stage)
            else:
                stage.rename(dst)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        print(f"  {action:9s}      {name}" + (f"   (basis written to {n} files)" if args.basis != "diagnostics" else ""))
        tally["re-imported" if action == "re-import" else "copied"] += 1

    print("\n" + "  ".join(f"{k}: {v}" for k, v in tally.items()))
    if args.dry_run:
        print("(dry run: nothing was copied)")


if __name__ == "__main__":
    sys.exit(main())
