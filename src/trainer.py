"""Shared training loop for the MLM and contrastive stages.

Built for an unattended overnight run on a remote machine:
  - checkpoints `ckpt_last.pt` (model + optimizer + scheduler + RNG state)
    every epoch and `ckpt_best.pt` whenever the monitored metric improves;
  - `--resume` restores all of that and continues from the next epoch;
  - a hard `time_budget_hours` stop that always saves before exiting;
  - JSONL metrics logging + a rendered loss-curve PNG at the end;
  - writes a DONE marker on graceful completion so the orchestrator can tell
    "finished" apart from "crashed" and retry with --resume.
"""

import json
import math
import random
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def warmup_cosine_schedule(optimizer, total_steps: int, warmup_frac: float = 0.1, min_lr_frac: float = 0.05):
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        run_dir: str | Path,
        device: torch.device | None = None,
        grad_clip: float = 1.0,
        log_every: int = 20,
        time_budget_hours: float | None = None,
        wandb_run=None,
        extra_state: dict | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or pick_device()
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.time_budget_s = time_budget_hours * 3600 if time_budget_hours else None
        self.wandb_run = wandb_run
        # Extra entries stored inside every checkpoint (e.g. model config,
        # tokenizer path) so a checkpoint is self-describing.
        self.extra_state = extra_state or {}

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

        self.model.to(self.device)
        self.use_amp = self.device.type == "cuda"

        self.epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.epochs_no_improve = 0
        self.history: list[dict] = []
        self._start_time = time.time()

    # ------------------------------------------------------------------ utils

    def _log(self, record: dict) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log({k: v for k, v in record.items() if isinstance(v, (int, float))})

    def _out_of_time(self) -> bool:
        return self.time_budget_s is not None and (time.time() - self._start_time) > self.time_budget_s

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self, name: str) -> None:
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "epochs_no_improve": self.epochs_no_improve,
            "history": self.history,
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            **self.extra_state,
        }
        tmp = self.run_dir / f"{name}.tmp"
        torch.save(payload, tmp)
        tmp.replace(self.run_dir / f"{name}.pt")  # atomic: no corrupt ckpt on crash

    def load_checkpoint(self, name: str = "ckpt_last") -> None:
        payload = torch.load(self.run_dir / f"{name}.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler and payload["scheduler_state"]:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        self.epoch = payload["epoch"]
        self.global_step = payload["global_step"]
        self.best_metric = payload["best_metric"]
        self.epochs_no_improve = payload["epochs_no_improve"]
        self.history = payload["history"]
        rng = payload["rng"]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        print(f"[trainer] resumed from {name}.pt at epoch {self.epoch}, step {self.global_step}")

    # ------------------------------------------------------------------- fit

    def fit(
        self,
        train_loader,
        compute_loss,
        epochs: int,
        validate=None,
        monitor: str | None = None,
        mode: str = "min",
        patience: int | None = None,
        resume: bool = False,
    ) -> None:
        """Run the training loop.

        compute_loss(model, batch) -> (loss, metrics_dict)
        validate(model) -> metrics_dict, must contain `monitor` if monitoring.
        """
        if resume and (self.run_dir / "ckpt_last.pt").exists():
            self.load_checkpoint("ckpt_last")
        start_epoch = self.epoch

        stop_reason = None
        for epoch in range(start_epoch, epochs):
            self.epoch = epoch
            self.model.train()
            running, n_batches = 0.0, 0
            print(f"[trainer] epoch {epoch + 1}/{epochs} started", flush=True)
            # disable=None → progress bar only on a real terminal, so piping
            # the overnight run into run.log doesn't fill it with \r frames.
            pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False, disable=None)
            for batch in pbar:
                batch = to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, metrics = compute_loss(self.model, batch)
                else:
                    loss, metrics = compute_loss(self.model, batch)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                self.global_step += 1
                loss_val = loss.item()
                if not math.isfinite(loss_val):
                    self.save_checkpoint("ckpt_nan")
                    raise RuntimeError(
                        f"non-finite loss at step {self.global_step} — "
                        "lower the learning rate (try --lr 1e-4) and restart with --resume"
                    )
                running += loss_val
                n_batches += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")

                if self.global_step % self.log_every == 0:
                    self._log({
                        "type": "step",
                        "step": self.global_step,
                        "epoch": epoch,
                        "loss": loss_val,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        **metrics,
                    })
                # Mid-epoch time check so a long epoch can't blow the budget.
                if self.global_step % 100 == 0 and self._out_of_time():
                    stop_reason = "time budget"
                    break

            epoch_record = {
                "type": "epoch",
                "epoch": epoch,
                "step": self.global_step,
                "train_loss": running / max(1, n_batches),
                "elapsed_min": (time.time() - self._start_time) / 60,
            }

            if validate is not None and stop_reason is None:
                self.model.eval()
                with torch.no_grad():
                    epoch_record.update(validate(self.model))

            self.history.append(epoch_record)
            self._log(epoch_record)
            self.epoch = epoch + 1  # checkpoints resume from the NEXT epoch
            print(f"[trainer] {json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in epoch_record.items()})}")

            if monitor and monitor in epoch_record:
                value = epoch_record[monitor]
                improved = (
                    self.best_metric is None
                    or (mode == "min" and value < self.best_metric)
                    or (mode == "max" and value > self.best_metric)
                )
                if improved:
                    self.best_metric = value
                    self.epochs_no_improve = 0
                    self.save_checkpoint("ckpt_best")
                    print(f"[trainer] new best {monitor}={value:.5f} → ckpt_best.pt")
                else:
                    self.epochs_no_improve += 1

            self.save_checkpoint("ckpt_last")

            if stop_reason:
                break
            if patience and self.epochs_no_improve >= patience:
                stop_reason = f"early stop ({patience} epochs without {monitor} improvement)"
                break
            if self._out_of_time():
                stop_reason = "time budget"
                break

        self._finish(stop_reason or "all epochs completed")

    def _finish(self, reason: str) -> None:
        if not (self.run_dir / "ckpt_best.pt").exists() and (self.run_dir / "ckpt_last.pt").exists():
            # No monitored metric (or no validation) — best := last.
            import shutil

            shutil.copyfile(self.run_dir / "ckpt_last.pt", self.run_dir / "ckpt_best.pt")
        with open(self.run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        self._plot_curves()
        (self.run_dir / "DONE").write_text(reason, encoding="utf-8")
        print(f"[trainer] finished: {reason} ({(time.time() - self._start_time) / 60:.1f} min)")

    def _plot_curves(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = [h["epoch"] + 1 for h in self.history]
            if not epochs:
                return
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, [h["train_loss"] for h in self.history], label="train loss", marker="o")
            if any("val_loss" in h for h in self.history):
                ax.plot(epochs, [h.get("val_loss") for h in self.history], label="val loss", marker="o")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend(loc="upper right")
            extra_keys = [k for k in self.history[-1] if k.startswith("val_") and k != "val_loss"]
            if extra_keys:
                ax2 = ax.twinx()
                for k in extra_keys:
                    ax2.plot(epochs, [h.get(k) for h in self.history], label=k, linestyle="--")
                ax2.set_ylabel("validation metric")
                ax2.legend(loc="center right")
            fig.tight_layout()
            fig.savefig(self.run_dir / "curves.png", dpi=120)
            plt.close(fig)
        except Exception as e:  # plotting must never kill an overnight run
            print(f"[trainer] curve plotting failed (non-fatal): {e}")
