"""Unattended overnight pipeline: tokenizer → MLM warm-up → contrastive → eval.

Designed to be launched once and left alone:
  - every stage is skipped if its DONE marker / artifact already exists, so
    re-running this script simply continues where it left off;
  - a crashed training stage is retried once with --resume (checkpoints are
    saved every epoch);
  - the total time budget is divided between stages and enforced inside the
    trainers, so the pipeline ALWAYS finishes within budget with whatever
    best checkpoint it has;
  - all output is mirrored to overnight.log next to this file's repo root.

Usage (PowerShell or bash, from the repo root):
    python scripts/run_overnight.py --time-budget-hours 10
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "overnight.log"

# Fraction of the *total* budget reserved for each training stage. MLM gets
# the bigger share; encoding + eval at the end needs only a few minutes, but
# we keep a 0.5h reserve so it can never be squeezed out.
MLM_FRACTION = 0.55
EVAL_RESERVE_HOURS = 0.5


def log(msg: str) -> None:
    line = f"[overnight {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str]) -> int:
    """Run a child process, mirroring its output to console + overnight.log."""
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
    return proc.wait()


def run_stage(name: str, cmd: list[str], done_marker: Path, resume_flag: bool = True) -> bool:
    if done_marker.exists():
        log(f"stage '{name}' already complete ({done_marker}) — skipping")
        return True
    start = time.time()
    code = run(cmd)
    if code != 0 and resume_flag:
        log(f"stage '{name}' exited with code {code} — retrying once with --resume")
        code = run(cmd + ["--resume"])
    elapsed = (time.time() - start) / 60
    log(f"stage '{name}' {'OK' if code == 0 else f'FAILED (code {code})'} after {elapsed:.1f} min")
    return code == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-budget-hours", type=float, default=10.0)
    parser.add_argument("--skip-mlm", action="store_true", help="contrastive-only run (not recommended)")
    parser.add_argument("--mlm-epochs", type=int, default=40)
    parser.add_argument("--contrastive-epochs", type=int, default=40)
    args = parser.parse_args()

    py = sys.executable
    deadline = time.time() + args.time_budget_hours * 3600
    remaining = lambda: max(0.0, (deadline - time.time()) / 3600)
    log(f"=== overnight run started, budget {args.time_budget_hours}h ===")

    # 0) sanity suite — refuses to launch a broken build into a long run
    if run([py, "scripts/sanity_checks.py"]) != 0:
        log("sanity checks FAILED — aborting before any training")
        sys.exit(1)

    # 1) tokenizer (idempotent: skips itself if tokenizer.json exists)
    if not run_stage("tokenizer", [py, "scripts/train_tokenizer.py"],
                     ROOT / "models/tokenizer_scratch/tokenizer.json", resume_flag=False):
        sys.exit(1)

    # 2) MLM warm-up
    if not args.skip_mlm:
        mlm_budget = min(args.time_budget_hours * MLM_FRACTION, remaining() - EVAL_RESERVE_HOURS)
        run_stage(
            "mlm",
            [py, "scripts/pretrain_mlm.py", "--epochs", str(args.mlm_epochs),
             "--time-budget-hours", f"{mlm_budget:.2f}"],
            ROOT / "runs/mlm/DONE",
        )
        # A failed MLM stage is not fatal: contrastive falls back to random init.

    # 3) contrastive training
    contrastive_budget = remaining() - EVAL_RESERVE_HOURS
    if contrastive_budget <= 0.1:
        log("no time left for contrastive training — increase the budget")
        sys.exit(1)
    if not run_stage(
        "contrastive",
        [py, "scripts/train_contrastive.py", "--epochs", str(args.contrastive_epochs),
         "--time-budget-hours", f"{contrastive_budget:.2f}"],
        ROOT / "runs/contrastive/DONE",
    ):
        sys.exit(1)

    # 4) encode corpora + test-set evaluation + demo
    code = run([py, "scripts/encode_and_eval.py"])
    log("=== overnight run finished ===")
    log("morning checklist — push/upload these:")
    log("  runs/mlm/history.json, runs/mlm/curves.png")
    log("  runs/contrastive/history.json, runs/contrastive/curves.png")
    log("  runs/contrastive/ckpt_best.pt   (large file: Drive or GitHub release)")
    log("  models/scratch/                 (embeddings + ids)")
    log("  evaluation/scratch_results.json")
    log("  overnight.log")
    sys.exit(code)


if __name__ == "__main__":
    main()
