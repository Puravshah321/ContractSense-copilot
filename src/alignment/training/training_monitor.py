"""
training_monitor.py
Logs training metrics: DPO loss, learning rate, step-wise performance.
Saves logs to file and optionally plots loss curve for reports.
"""

import json
import time
from pathlib import Path
from typing import Optional


class TrainingMonitor:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.logs = {
            "start_time": None,
            "end_time": None,
            "total_steps": 0,
            "steps": [],
            "epoch_metrics": [],
        }
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def start(self):
        self.logs["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"📊 Training monitor started at {self.logs['start_time']}")

    def log_step(self, step: int, loss: float, lr: float, epoch: float = 0.0):
        self.logs["steps"].append({
            "step": step,
            "loss": loss,
            "learning_rate": lr,
            "epoch": epoch,
            "timestamp": time.strftime("%H:%M:%S"),
        })
        self.logs["total_steps"] = step

    def log_epoch(self, epoch: int, train_loss: float, eval_loss: float,
                  dpo_rewards_chosen: float = 0.0, dpo_rewards_rejected: float = 0.0):
        self.logs["epoch_metrics"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "dpo_rewards_chosen": dpo_rewards_chosen,
            "dpo_rewards_rejected": dpo_rewards_rejected,
            "reward_margin": dpo_rewards_chosen - dpo_rewards_rejected,
        })
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}, "
              f"margin={dpo_rewards_chosen - dpo_rewards_rejected:.4f}")

    def finish(self):
        self.logs["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        print(f"📊 Training monitor finished at {self.logs['end_time']}")

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"💾 Training logs saved to {self.log_path}")

    def plot_loss_curve(self, output_path: Optional[str] = None):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not self.logs["steps"]:
                print("⚠️  No step data to plot")
                return

            steps = [s["step"] for s in self.logs["steps"]]
            losses = [s["loss"] for s in self.logs["steps"]]

            fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

            ax1.plot(steps, losses, color="#4A90D9", linewidth=1.5, alpha=0.8)
            ax1.set_xlabel("Training Step")
            ax1.set_ylabel("DPO Loss")
            ax1.set_title("DPO Training Loss Curve")
            ax1.grid(True, alpha=0.3)

            window = max(1, len(losses) // 50)
            if window > 1:
                smoothed = []
                for i in range(len(losses)):
                    start = max(0, i - window)
                    smoothed.append(sum(losses[start:i+1]) / (i - start + 1))
                ax1.plot(steps, smoothed, color="#E74C3C", linewidth=2, label="Smoothed")
                ax1.legend()

            plt.tight_layout()

            if output_path is None:
                output_path = str(Path(self.log_path).parent / "training_loss_curve.png")

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"📈 Loss curve saved to {output_path}")

        except ImportError:
            print("⚠️  matplotlib not available, skipping plot")

    def plot_epoch_metrics(self, output_path: Optional[str] = None):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not self.logs["epoch_metrics"]:
                print("⚠️  No epoch data to plot")
                return

            epochs = [e["epoch"] for e in self.logs["epoch_metrics"]]
            train_losses = [e["train_loss"] for e in self.logs["epoch_metrics"]]
            eval_losses = [e["eval_loss"] for e in self.logs["epoch_metrics"]]
            margins = [e["reward_margin"] for e in self.logs["epoch_metrics"]]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(epochs, train_losses, "o-", color="#4A90D9", label="Train Loss")
            ax1.plot(epochs, eval_losses, "o-", color="#E74C3C", label="Eval Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Train vs Eval Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.bar(epochs, margins, color="#2ECC71", alpha=0.8)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Reward Margin")
            ax2.set_title("DPO Reward Margin (Chosen - Rejected)")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if output_path is None:
                output_path = str(Path(self.log_path).parent / "epoch_metrics.png")

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"📈 Epoch metrics plot saved to {output_path}")

        except ImportError:
            print("⚠️  matplotlib not available, skipping plot")


def parse_trainer_log(trainer_state_path: str) -> list:
    with open(trainer_state_path) as f:
        state = json.load(f)

    return state.get("log_history", [])


def build_monitor_from_trainer_state(trainer_state_path: str, log_output_path: str) -> TrainingMonitor:
    monitor = TrainingMonitor(log_output_path)
    log_history = parse_trainer_log(trainer_state_path)

    monitor.start()

    current_epoch = 0
    for entry in log_history:
        step = entry.get("step", 0)

        if "loss" in entry:
            monitor.log_step(
                step=step,
                loss=entry["loss"],
                lr=entry.get("learning_rate", 0),
                epoch=entry.get("epoch", 0),
            )

        if "eval_loss" in entry:
            current_epoch += 1
            monitor.log_epoch(
                epoch=current_epoch,
                train_loss=entry.get("loss", entry.get("train_loss", 0)),
                eval_loss=entry["eval_loss"],
                dpo_rewards_chosen=entry.get("rewards/chosen", 0),
                dpo_rewards_rejected=entry.get("rewards/rejected", 0),
            )

    monitor.finish()
    return monitor
