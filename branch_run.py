#!/usr/bin/env python
"""
branch_run.py
=============
Continue an existing run, of either basis, in a NEW data folder that points
back at the old one.

Takes a data folder (a sweep folder, or a run folder directly) plus a run number,
reloads that run's `config.json`, optionally patches it, seeds the new folder with
the parent's checkpoint, and hands the whole thing to its basis's entry point
(spinbasis/run_2D_spinbasis.py or eigenbasis/run_2D_eigenbasis.py) for a
given number of further iteration steps. The parent is never written to.

WHY A NEW FOLDER
----------------
run.sh already resumes a run IN PLACE: it re-runs the driver against the same
directory, the driver sees run.mpack, reads run.progress, and continues. That is
the right thing for "this job hit the wall clock, give it another window", and
this script is not a replacement for it.

What it is for is the other case: you want to take a converged (or
half-converged) state and push it somewhere new -- more steps, a different
learning rate, symmetries switched on, a different J2 -- WITHOUT destroying the
run you already paid for. Resuming in place would overwrite run.mpack and
append to the same logs, so the original state would be gone. Branching copies
the checkpoint into a fresh folder and leaves the parent untouched.

HOW THE CONTINUATION WORKS
--------------------------
The driver's resume contract (run_2D_eigenbasis.py) is entirely file-based:

  * `<dir>/run.mpack` exists            -> do_resume = True
  * `<dir>/run.progress` -> cum_steps   -> step_offset  (and `alpha`, which lives
                                           outside the checkpoint, for auto_alpha)
  * `model.iter` is the TOTAL target    -> remaining = iter - step_offset
  * the lr / diag_shift schedules are offset-aware, so they continue from the
    global step instead of re-annealing

So a branch is just: a new directory holding a copy of the parent's `run.mpack`
and `run.progress`, a `config.json` whose `model.iter` is
`parent_steps + --extra`, and then the ordinary driver invocation
`python run_2D_eigenbasis.py <dir>/config.json <dir>`.

THE ONE REAL FOOTGUN
--------------------
`run.mpack` is `flax.serialization.to_bytes(vstate.variables)` -- the parameter
tree and nothing else. It can only be restored into an IDENTICALLY SHAPED model.
Any override that changes the architecture (layers, ratio, heads,
attention_mode, embedding, L, transl_invariant) or wraps it in the C4v
projection (symmetry.elements, symmetry.spin_flip, which move the network under
a submodule) makes the checkpoint unrestorable. This script refuses such a
branch unless you pass --fresh, which starts from a random initialisation while
keeping the provenance link.

Changing the HAMILTONIAN (J1, J2, sign_rule) does restore fine -- same parameter
tree -- and is a legitimate and useful branch: carrying a J2 = 0 solution into
J2 > 0 is exactly the transfer-learning trick. It is a different physical
problem, though, so the script warns and records it.

LAYOUT IT WRITES
----------------
The new folder mimics a one-run sweep, so the existing notebooks
(`plot_energy_vs_iteration.ipynb`, `ANALYSIS.ipynb`, `plot_depth_vs_width.ipynb`)
pick it up with no changes:

    <out-root>/<stamp>_branch_<parent-stamp>_run<N>/
        matrix.json        one-run sweep matrix, with a `parent` block
        parent.json        full provenance pointer to the source run
        run_0000/
            config.json    the patched config the driver is given
            manifest.json  as run.sh writes, plus the same `parent` block
            parent.json    same pointer, so a run dir alone is self-describing
            run.mpack      copied from the parent  (unless --fresh)
            run.progress   copied from the parent  (unless --fresh)
            run.log, run step *.log, run.diag.jsonl
                           copied from the parent, so plots show one continuous
                           curve from step 0 (skipped by --no-history, and by
                           --fresh, which starts its own history at step 0)
            stdout.txt, stderr.txt
                           this run's output, as run.sh does it

SUBMITTING IT
-------------
`run.sh` takes `(matrix.json, sweep_dir)` and reads the run index from
`SLURM_ARRAY_TASK_ID`, so a one-run branch folder is just `--array=0`. The
resource flags come from `clusters/<name>.sh` on the sbatch command line, and
the profile is pinned per sweep in `<sweep_dir>/cluster.txt`, which this script
copies from the parent. `--submit` does the whole thing; without it the exact
command is printed for you to paste.

Because `run.sh` regenerates `config.json` from `matrix.json` and resubmits
itself with the same `(matrix, sweep_dir)`, a branch picks up the automatic
continue-after-timeout behaviour with no extra work. It also rewrites
`manifest.json`, which is why the provenance is kept in `parent.json` instead of
only in the manifest.

EXAMPLES
--------
    # 5000 more steps from run_0002 of a finished sweep
    python branch_run.py --from data/eigenbasis/2026-10-02_19-43-18G3352432P4193954 \
                         --run 2 --extra 5000

    # same, but drop the SR step size for the refinement phase
    python branch_run.py --from data/eigenbasis/2026-10-02_... --run 2 --extra 5000 \
        --set optimizer.initial_learning_rate=0.01 \
        --set optimizer.final_learning_rate=0.005

    # staged protocol: switch the point group on for a short final phase.
    # changes the parameter tree, so it cannot reuse the checkpoint
    python branch_run.py --from data/eigenbasis/2026-10-02_... --run 2 --extra 500 \
        --set 'symmetry.elements=["I","C2","sigma_d","sigma_a"]' --fresh

    # on AICR: prepare and submit as a one-task array
    python branch_run.py --from data/eigenbasis/2026-10-02_... --run 2 --extra 5000 --submit
"""

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def basis_of(cfg):
    """[system] basis of a config. Configs written before the field existed
    are inferred: only the spin-basis schema has [model] type."""
    b = cfg.get("system", {}).get("basis")
    if b is None:
        b = "spin" if "type" in cfg.get("model", {}) else "eigen"
    if b not in ("spin", "eigen"):
        sys.exit(f"ERROR: unknown [system] basis {b!r} in the parent config")
    return b


def driver_for(basis):
    return Path(f"{basis}basis") / f"run_2D_{basis}basis.py"


# Config keys that determine the SHAPE of vstate.variables. A branch that
# changes any of these cannot restore the parent's run.mpack (see the module
# docstring), so it requires --fresh. Eigenbasis keys are listed; spin-basis
# keys are the whole selected [model.<type>] table (see arch_keys).
ARCH_KEYS = (
    "system.L",
    "system.dimensions",
    "system.transl_invariant",
    "model.layers",
    "model.ratio",            # d_model = ratio * heads
    "model.heads",
    "model.attention_mode",
    "model.embedding",
    "symmetry.elements",      # non-empty => wrapped in SymmProject, new tree
    "symmetry.spin_flip",
)

def arch_keys(basis, parent_cfg, new_cfg):
    if basis == "eigen":
        return ARCH_KEYS
    keys = {"system.L", "model.type"}
    for cfg in (parent_cfg, new_cfg):
        kind = cfg.get("model", {}).get("type")
        for k in (cfg.get("model", {}).get(kind) or {}):
            keys.add(f"model.{kind}.{k}")
    return tuple(sorted(keys))


# Keys that change the physics but not the parameter tree: the checkpoint
# restores, but the run is solving a different Hamiltonian than its parent.
PHYSICS_KEYS = ("system.J1", "system.J2", "system.sign_rule")


# ---------------------------------------------------------------------------
# dotted-path helpers
# ---------------------------------------------------------------------------
_MISSING = object()


def dget(cfg, path, default=_MISSING):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def dset(cfg, path, value):
    """Set a dotted path. The top-level section must already exist -- that is
    what catches `--set optimiser.lr` (wrong section) -- but a new leaf inside a
    known section is allowed, since optional blocks like [symmetry] and keys
    like distribution.auto_alpha are simply absent from older configs."""
    parts = path.split(".")
    if parts[0] not in cfg:
        raise KeyError(f"no section {parts[0]!r} in the parent config "
                       f"(sections: {', '.join(sorted(cfg))})")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if nxt is None:
            nxt = cur[part] = {}
        elif not isinstance(nxt, dict):
            raise KeyError(f"cannot descend into {part!r} while setting {path!r}: "
                           f"it holds {type(nxt).__name__}, not a table")
        cur = nxt
    existed = parts[-1] in cur
    cur[parts[-1]] = value
    return existed


def parse_value(text):
    """JSON first, so numbers / bools / lists keep their types; bare words
    (e.g. sign_rule=TF, attention_mode=factored) fall back to str."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ---------------------------------------------------------------------------
# locating the parent
# ---------------------------------------------------------------------------
def resolve_parent(src, run_id):
    """(sweep_dir_or_None, run_dir). `src` may be a sweep folder or a run folder."""
    src = Path(src).expanduser().resolve()
    if not src.is_dir():
        sys.exit(f"ERROR: --from is not a directory: {src}")

    if re.fullmatch(r"run_\d{4}", src.name):
        if run_id is not None:
            parsed = int(src.name.split("_")[1])
            if parsed != run_id:
                sys.exit(f"ERROR: --from points at {src.name} but --run {run_id} "
                         f"was given; drop --run or pass the sweep folder.")
        return (src.parent if (src.parent / "matrix.json").exists() else None), src

    rid = 0 if run_id is None else run_id
    run_dir = src / f"run_{rid:04d}"
    if not run_dir.is_dir():
        have = sorted(p.name for p in src.glob("run_*"))
        sys.exit(f"ERROR: no {run_dir.name} under {src}"
                 + (f"\n       available: {', '.join(have)}" if have else ""))
    return src, run_dir


def load_parent_config(run_dir, sweep_dir, run_id):
    """config.json is what the driver was actually given, so prefer it. Fall back
    to manifest.json and then to the sweep matrix, which are the only sources
    left for a run whose directory was never populated."""
    p = run_dir / "config.json"
    if p.exists():
        return json.loads(p.read_text()), str(p)

    p = run_dir / "manifest.json"
    if p.exists():
        man = json.loads(p.read_text())
        if isinstance(man.get("config"), dict):
            return man["config"], f"{p} (config block)"

    if sweep_dir is not None:
        p = sweep_dir / "matrix.json"
        if p.exists():
            for spec in json.loads(p.read_text()).get("runs", []):
                if int(spec.get("run_id", -1)) == run_id:
                    return spec["config"], f"{p} (run {run_id})"

    sys.exit(f"ERROR: no config found for {run_dir}\n"
             f"       looked for config.json, manifest.json, "
             f"and the sweep matrix.json")


def parent_steps(run_dir):
    """Steps the parent actually applied. run.progress is authoritative -- the
    driver writes it next to every checkpoint -- with the logs as a fallback for
    a run whose .progress went missing."""
    p = run_dir / "run.progress"
    if p.exists():
        try:
            return int(json.loads(p.read_text()).get("cum_steps", 0)), "run.progress"
        except Exception:
            pass
    best = -1
    for lp in run_dir.glob("*.log"):
        try:
            its = json.loads(lp.read_text())["Energy"]["iters"]
            best = max(best, int(its[-1]))
        except Exception:
            continue
    if best >= 0:
        return best + 1, "last iteration in *.log"
    return 0, "nothing found (treated as a fresh start)"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def git_commit(repo):
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Continue a run (either basis) in a new data folder that points "
                    "back at the old one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for the resume contract and examples.")
    ap.add_argument("--from", dest="src", required=True, metavar="DIR",
                    help="sweep folder (with --run N) or a run_XXXX folder")
    ap.add_argument("--run", type=int, default=None, metavar="N",
                    help="run number inside the sweep folder (default 0)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--extra", type=int, metavar="N",
                   help="run N MORE steps than the parent reached")
    g.add_argument("--total", type=int, metavar="N",
                   help="set the total step target to N (must exceed the parent's)")
    ap.add_argument("--out-root", default=None, metavar="DIR",
                    help="where to create the branch folder "
                         "(default: the parent sweep's parent directory)")
    ap.add_argument("--name", default=None,
                    help="branch folder name (default: <stamp>_branch_<parent>_run<N>)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="PATH=VALUE",
                    help="patch the config, e.g. optimizer.final_learning_rate=0.005 "
                         "(repeatable; VALUE is parsed as JSON, else kept as a string)")
    ap.add_argument("--fresh", action="store_true",
                    help="do NOT seed the parent's checkpoint: start from a random "
                         "init, keeping only the config and the provenance link. "
                         "Required when an override changes the parameter tree.")
    ap.add_argument("--no-history", action="store_true",
                    help="do not copy the parent's logs / diagnostics, so the "
                         "branch folder holds only the steps it ran itself. "
                         "Implied by --fresh, which starts at step 0.")
    run_mode = ap.add_mutually_exclusive_group()
    run_mode.add_argument("--submit", action="store_true",
                          help="sbatch the branch as a one-task array via run.sh, "
                               "using the cluster profile's resource flags "
                               "(this is what you want on AICR)")
    run_mode.add_argument("--run-local", action="store_true",
                          help="run the driver here, in the foreground "
                               "(an interactive/compute node, not a login node)")
    ap.add_argument("--dry-run", action="store_true",
                    help="explicit no-op: preparing the folder and printing the "
                         "commands is already the default")
    ap.add_argument("--cluster", default=None, metavar="NAME",
                    help="cluster profile to pin in the branch's cluster.txt "
                         "(default: inherit the parent sweep's; run.sh otherwise "
                         "falls back to autodetection)")
    ap.add_argument("--python", default=None, metavar="PY",
                    help="interpreter for --run-local (default: ./.venv/bin/python "
                         "if present, else the current one)")
    ap.add_argument("--emit-dir", default=None, metavar="FILE",
                    help="write the branch folder's absolute path to FILE, for "
                         "a calling script to pick up (submit_branch.sh uses this)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent

    sweep_dir, run_dir = resolve_parent(args.src, args.run)
    run_id = int(run_dir.name.split("_")[1])
    cfg, cfg_src = load_parent_config(run_dir, sweep_dir, run_id)
    basis = basis_of(cfg)
    cfg.setdefault("system", {})["basis"] = basis
    driver = repo / driver_for(basis)
    if not driver.exists():
        sys.exit(f"ERROR: {driver_for(basis)} not found under this script ({repo})")
    parent_cfg = copy.deepcopy(cfg)
    steps, steps_src = parent_steps(run_dir)

    ckpt = run_dir / "run.mpack"
    seed_ckpt = not args.fresh
    if seed_ckpt and not ckpt.exists():
        sys.exit(f"ERROR: parent has no checkpoint to continue from:\n"
                 f"       {ckpt}\n"
                 f"       pass --fresh to branch the config only.")

    # ---- step target -----------------------------------------------------
    base = 0 if args.fresh else steps
    if args.extra is not None:
        total = base + args.extra
    else:
        total = args.total
        if total <= base:
            sys.exit(f"ERROR: --total {total} is not beyond the parent's "
                     f"{base} applied steps; nothing would run.")
    new_cfg = copy.deepcopy(cfg)
    new_cfg["model"]["iter"] = int(total)

    # ---- overrides -------------------------------------------------------
    applied = {}
    for item in args.overrides:
        if "=" not in item:
            sys.exit(f"ERROR: --set wants PATH=VALUE, got {item!r}")
        path, _, raw = item.partition("=")
        path, value = path.strip(), parse_value(raw)
        try:
            existed = dset(new_cfg, path, value)
        except KeyError as e:
            sys.exit(f"ERROR: --set {path}: {e}")
        applied[path] = value
        if not existed:
            print(f"note: {path} was absent from the parent config; adding it")

    # Spin basis: keep only the selected [model.<type>] table, as submit.sh
    # does. Older parents carry all three; build_model reads only this one.
    if basis == "spin":
        kind = new_cfg["model"].get("type")
        if not isinstance(new_cfg["model"].get(kind), dict):
            sys.exit(f"ERROR: [model] type = {kind!r} but the config has no [model.{kind}] table")
        for k in [k for k, v in new_cfg["model"].items() if isinstance(v, dict) and k != kind]:
            del new_cfg["model"][k]

    # ---- the footgun check ----------------------------------------------
    arch_changed = [k for k in arch_keys(basis, parent_cfg, new_cfg)
                    if dget(parent_cfg, k, None) != dget(new_cfg, k, None)]
    if arch_changed and seed_ckpt:
        detail = "\n".join(f"         {k}: {dget(parent_cfg, k, None)!r} "
                           f"-> {dget(new_cfg, k, None)!r}" for k in arch_changed)
        sys.exit("ERROR: these overrides change the parameter tree, so the "
                 "parent's\n       run.mpack cannot be restored into the new "
                 f"model:\n{detail}\n"
                 "       re-run with --fresh to start from a random init "
                 "(provenance is still recorded).")

    physics_changed = [k for k in PHYSICS_KEYS
                       if dget(parent_cfg, k, None) != dget(new_cfg, k, None)]

    # ---- destination -----------------------------------------------------
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_stamp = (sweep_dir.name if sweep_dir is not None else run_dir.parent.name)[:19]
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else (
        sweep_dir.parent if sweep_dir is not None else run_dir.parent.parent)
    name = args.name or f"{stamp}_branch_{parent_stamp}_run{run_id}"
    branch = out_root / name
    new_run = branch / "run_0000"
    if branch.exists():
        sys.exit(f"ERROR: destination already exists: {branch}")
    new_run.mkdir(parents=True)

    # ---- provenance ------------------------------------------------------
    prov = {
        "branched_at": datetime.now().isoformat(timespec="seconds"),
        "branch_tool": Path(__file__).name,
        "repo_git_commit": git_commit(repo),
        "parent": {
            "sweep_id": (json.loads((sweep_dir / "matrix.json").read_text()).get("sweep_id")
                         if sweep_dir is not None and (sweep_dir / "matrix.json").exists()
                         else None),
            "sweep_dir": str(sweep_dir) if sweep_dir is not None else None,
            "run_dir": str(run_dir),
            "run_id": run_id,
            "config_source": cfg_src,
            "applied_steps": steps,
            "applied_steps_source": steps_src,
            "git_commit": (json.loads((run_dir / "manifest.json").read_text()).get("git_commit")
                           if (run_dir / "manifest.json").exists() else None),
            "checkpoint": {
                "path": str(ckpt) if ckpt.exists() else None,
                "sha256": sha256(ckpt) if ckpt.exists() else None,
                "bytes": ckpt.stat().st_size if ckpt.exists() else None,
                "reused": bool(seed_ckpt),
            },
        },
        "branch": {
            "start_step": base,
            "target_step": int(total),
            "steps_to_run": int(total) - base,
            "fresh_init": bool(args.fresh),
            "history_copied": bool(seed_ckpt and not args.no_history),
            "overrides": applied,
            "arch_changed": arch_changed,
            "physics_changed": physics_changed,
        },
    }

    (new_run / "config.json").write_text(json.dumps(new_cfg, indent=2) + "\n")
    for d in (branch, new_run):
        (d / "parent.json").write_text(json.dumps(prov, indent=2) + "\n")

    # manifest.json in the shape run.sh writes, plus the provenance block, so
    # ANALYSIS.ipynb's loader finds the config where it expects it
    manifest = {
        "sweep_id": name,
        "run_id": 0,
        "basis": basis,
        "git_commit": prov["repo_git_commit"],
        "submitted_config": str(new_run / "config.json"),
        "sweep_parameters": [],
        "sweep_values": applied,
        "config": new_cfg,
        "parent": prov["parent"],
        "branch": prov["branch"],
    }
    (new_run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # one-run sweep matrix, so the plotting notebooks treat this like any sweep
    (branch / "matrix.json").write_text(json.dumps({
        "sweep_id": name,
        "basis": basis,
        "git_commit": prov["repo_git_commit"],
        "source_config": str(new_run / "config.json"),
        "sweep_parameters": [],
        "sweep_zip_parameters": list(applied),
        "num_runs": 1,
        "parent": prov["parent"],
        "runs": [{"run_id": 0, "sweep_values": applied, "config": new_cfg}],
    }, indent=2) + "\n")

    # ---- cluster pin -----------------------------------------------------
    # run.sh resolves the cluster profile from $SWEEP_DIR/cluster.txt and only
    # falls back to autodetection (hostname / partition regexes) if it is
    # absent -- and hard-errors if that also fails. Carry the parent's pin over
    # so a branch submits with the same resources it was produced with.
    cluster = args.cluster
    if cluster is None and sweep_dir is not None:
        src_cluster = sweep_dir / "cluster.txt"
        if src_cluster.exists():
            cluster = src_cluster.read_text().strip()
    if cluster:
        (branch / "cluster.txt").write_text(cluster + "\n")

    # submit.sh enforces that MAX_RUNTIME_MIN stays under the profile's
    # walltime, because MAX_RUNTIME_MIN is what stops the run gracefully and
    # flushes the checkpoint; if it exceeds --time, Slurm hard-kills the job
    # first and the window's progress is lost. Branching does not go through
    # submit.sh, so repeat the check here.
    budget = dget(new_cfg, "run.MAX_RUNTIME_MIN", None)
    cap = None
    if cluster:
        prof = repo / "clusters" / f"{cluster}.sh"
        if prof.exists():
            mt = re.search(r"^CLUSTER_MAX_RUNTIME_MIN=(\d+)", prof.read_text(), re.M)
            cap = int(mt.group(1)) if mt else None

    # ---- seed the state --------------------------------------------------
    copied = []
    if seed_ckpt:
        shutil.copy2(ckpt, new_run / "run.mpack")
        copied.append("run.mpack")
        src_prog = run_dir / "run.progress"
        if src_prog.exists():
            shutil.copy2(src_prog, new_run / "run.progress")
        else:
            # the driver needs cum_steps to set step_offset and keep the lr /
            # diag_shift schedules on the global clock
            (new_run / "run.progress").write_text(
                json.dumps({"cum_steps": int(steps)}) + "\n")
        copied.append("run.progress")
    # History only makes sense alongside a reused checkpoint. On a --fresh
    # branch the driver is not resuming, so it opens run.log in mode='write'
    # (clobbering a copy) and deletes run.diag.jsonl outright -- and the steps
    # would not belong to this lineage anyway.
    copy_history = seed_ckpt and not args.no_history
    if copy_history:
        for src in sorted(run_dir.glob("*.log")) + [run_dir / "run.diag.jsonl"]:
            if src.exists():
                shutil.copy2(src, new_run / src.name)
                copied.append(src.name)

    if args.emit_dir:
        Path(args.emit_dir).write_text(str(branch) + "\n")

    # ---- report ----------------------------------------------------------
    py = args.python or (str(repo / ".venv" / "bin" / "python")
                         if (repo / ".venv" / "bin" / "python").exists()
                         else sys.executable)
    cmd = [py, "-u", str(driver), str(new_run / "config.json"), str(new_run)]

    print(f"\nparent : {run_dir}")
    print(f"         {steps} steps applied (from {steps_src})")
    print(f"branch : {new_run}")
    print(f"         {base} -> {total}  ({total - base} steps to run)"
          + ("   [fresh init]" if args.fresh else "   [checkpoint reused]"))
    if copied:
        print(f"seeded : {', '.join(copied)}")
    if applied:
        print("config overrides:")
        for k, v in applied.items():
            print(f"         {k}: {dget(parent_cfg, k, None)!r} -> {v!r}")
    if physics_changed:
        print("WARNING: this branch changes the Hamiltonian, so it is a different")
        print("         physical problem than its parent:")
        for k in physics_changed:
            print(f"         {k}: {dget(parent_cfg, k, None)!r} "
                  f"-> {dget(new_cfg, k, None)!r}")

    # The cluster route: run.sh takes (matrix.json, sweep_dir) and reads the run
    # index from SLURM_ARRAY_TASK_ID, so a one-run branch is --array=0. The
    # resource flags live in the cluster profile and must be on the command line
    # (run.sh carries only portable #SBATCH directives), which is also how
    # run.sh resubmits itself after a clean wall-clock stop.
    sbatch_sh = (f'cd {repo} && source clusters/common.sh && '
                 f'cluster_load {cluster or ""} && '
                 f'sbatch "${{CLUSTER_SBATCH_ARGS[@]}}" --array=0 '
                 f'{repo / "run.sh"} {branch / "matrix.json"} {branch}')

    print(f"cluster: {cluster or 'not pinned -- run.sh will autodetect'}"
          + (f"   (walltime budget {cap} min)" if cap else ""))
    if budget is not None and cap is not None and budget > cap:
        print(f"WARNING: run.MAX_RUNTIME_MIN = {budget} exceeds the {cluster} "
              f"profile's {cap} min.")
        print(f"         Slurm will hard-kill the job before the driver can "
              f"checkpoint, losing that window.")
        print(f"         Either --set run.MAX_RUNTIME_MIN={cap}, or submit with a "
              f"longer --time and a matching partition.")

    # Suppressed when a wrapper is driving us (submit_branch.sh passes
    # --emit-dir): it prints its own summary and submits, so offering a second,
    # competing sbatch line and "nothing launched" would just be confusing.
    if not args.emit_dir:
        print("\nto submit on the cluster (one array task):")
        print(f"  {sbatch_sh}")
        print("\nto run it here instead, in the foreground:")
        print(f"  {' '.join(cmd)}")

    if args.submit:
        print("\n--submit: handing off to sbatch ...", flush=True)
        return subprocess.run(["bash", "-lc", sbatch_sh], cwd=str(repo)).returncode

    if not args.run_local:
        if not args.emit_dir:
            print("\nfolder prepared; nothing launched "
                  "(pass --submit, or --run-local).")
        return 0

    # Same stream split as run.sh: the driver's tqdm bar goes to stderr and is
    # enormous, so it is filed rather than echoed, while stdout is tee'd.
    with open(new_run / "stdout.txt", "a") as out, \
         open(new_run / "stderr.txt", "a") as err:
        proc = subprocess.Popen(cmd, cwd=str(repo), stdout=subprocess.PIPE,
                                stderr=err, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            out.write(line)
        rc = proc.wait()

    print(f"\ndriver exited {rc}")
    for flag, msg in (("run.done", "target reached (.done)"),
                      ("run.timeout", "clean wall-clock stop (.timeout) -- "
                                      "rerun the command above to continue"),):
        if (new_run / flag).exists():
            print(f"status : {msg}")
            break
    else:
        print("status : stopped without a sentinel (see stderr.txt)")
    print(f"outputs: {new_run}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
