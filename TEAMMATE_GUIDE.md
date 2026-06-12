# GPU Training Guide (Windows / RTX 5070)

You are running the training half of our NLP project. Everything is scripted —
the happy path is: **setup → verify (5 min) → calibrate (15 min) → launch the
overnight run → push results in the morning.** All commands are PowerShell,
run from the repo root.

If anything breaks, open this repo in Cursor or Codex and ask — `AGENTS.md`
gives the AI assistant full context about this project.

## 1. One-time setup

```powershell
git clone <REPO_URL>
cd Neural-Search-Engine
git checkout scratch-encoder

py -3.12 -m venv .venv          # any Python 3.10-3.13 works
.\.venv\Scripts\Activate.ps1

# Torch FIRST, from the CUDA 12.8 wheel index — the RTX 5070 is a Blackwell
# card (sm_120) and the default pip torch is too old for it:
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements-train.txt

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True NVIDIA GeForce RTX 5070`. If you see `False` or a
"no kernel image is available" error later, the torch install is wrong —
redo the `pip install --index-url ...` line and check it reports torch >= 2.7.

## 2. Verify the build (~5 min)

```powershell
python scripts\sanity_checks.py
```

All 5 checks must say `PASS`. Then a tiny end-to-end smoke run:

```powershell
python scripts\train_tokenizer.py
python scripts\pretrain_mlm.py --smoke
python scripts\train_contrastive.py --smoke --init-from runs\mlm_smoke\ckpt_best.pt
```

Each finishes in ~1-2 minutes. You're checking that they run to
"[trainer] finished", not the metric values (smoke metrics are meaningless).

## 3. Calibration (~15 min) — do this while we're both online

```powershell
python scripts\pretrain_mlm.py --epochs 1 --run-dir runs\calib_mlm
python scripts\train_contrastive.py --epochs 1 --no-init --val-queries 200 --run-dir runs\calib_contrastive
```

Send me the `[trainer] {...}` epoch lines. We're checking:
- loss decreases and is finite (if it NaNs, the script aborts and tells you the fix),
- minutes per epoch, so we can size the overnight epoch counts.

The `--run-dir runs\calib_*` part matters — it keeps calibration from
occupying the real run directories.

## 4. Overnight launch

```powershell
# keep the machine awake while plugged in (one-time):
powercfg /change standby-timeout-ac 0

python scripts\run_overnight.py --time-budget-hours 10
```

Leave the PowerShell window open (minimized is fine; the monitor can sleep).
The pipeline runs: sanity checks → tokenizer → MLM warm-up → contrastive
training → encoding + evaluation + demo, and writes everything it prints to
`overnight.log`.

Built-in safety nets — you do NOT need to watch it:
- checkpoints are saved every epoch; **if anything crashes or you must reboot,
  just re-run the exact same command** — finished stages are skipped and the
  interrupted stage resumes from its last checkpoint;
- the time budget is enforced internally: it will always finish within ~10h
  with the best checkpoint it reached;
- a crashed stage is automatically retried once with `--resume`.

Optional live monitoring from your browser: `pip install wandb`, `wandb login`,
then add `--wandb` to the stage commands (skip this if login is any trouble).

## 5. Morning checklist

1. Check the end of `overnight.log` — you should see the results table
   (BiEncoder-scratch vs TF-IDF vs BM25) and "overnight run finished".
2. Commit and push the small artifacts:
   ```powershell
   git add runs evaluation models/scratch models/tokenizer_scratch overnight.log
   git commit -m "Overnight training run results"
   git push
   ```
   (Checkpoints `*.pt` are gitignored — that's intended.)
3. Upload `runs\contrastive\ckpt_best.pt` (~230 MB) to Google Drive and send
   me the link.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no kernel image is available for execution` | torch is not the cu128 build — redo step 1's torch install |
| `CUDA out of memory` | re-run the stage with `--batch-size 128` (or 64) |
| non-finite loss (script aborts with a message) | delete that stage's run dir (`Remove-Item -Recurse -Force runs\mlm` or `runs\contrastive`) and re-run the stage with `--lr 1e-4`; then re-run `run_overnight.py` to continue the pipeline |
| machine rebooted overnight | just re-run `python scripts\run_overnight.py --time-budget-hours <hours left>` |
| anything else | open the repo in Cursor/Codex and paste the error — `AGENTS.md` briefs the AI on everything |
