"""
Unified evaluation and comparison of all segmentation methods.

Produces:
  - results/comparison_table.csv   — per-method mean ± std for all metrics
  - results/ablation_table.csv     — effect of labeled_ratio on semi-supervised
  - results/per_case_results.json  — raw per-case metrics for all methods
  - results/figures/               — box plots, bar charts, per-case scatter plots

Usage:
    python scripts/06_compare_methods.py --data_dir <path> --experiments_dir <path>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import PancreasDataset
from src.models.unet import build_model
from src.evaluation.metrics import evaluate_case, aggregate_metrics
from src.utils.io import load_checkpoint, load_config, save_results_json
from src.data.transforms import PATCH

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = {
    "supervised":       "#2196F3",
    "semisup_10pct":    "#4CAF50",
    "semisup_20pct":    "#8BC34A",
    "selfsup_finetune": "#FF9800",
    "fewshot_5shot":    "#9C27B0",
}
METRICS_TO_REPORT = [
    "tumor_dice", "tumor_hd95", "tumor_vol_error",
    "pancreas_dice", "pancreas_hd95",
]


# ── Inference helper ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, val_loader, device, patch_size, num_classes=3) -> list:
    """Run sliding-window inference and return per-case metric dicts."""
    model.eval()
    post_pred  = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)
    per_case   = []

    for batch in val_loader:
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = sliding_window_inference(
            inputs=imgs, roi_size=patch_size,
            sw_batch_size=2, predictor=model, overlap=0.5,
        )
        preds = [post_pred(p).argmax(0).cpu().numpy()  for p in decollate_batch(logits)]
        gts   = [post_label(l).argmax(0).cpu().numpy() for l in decollate_batch(labels)]

        for pred, gt in zip(preds, gts):
            per_case.append(evaluate_case(
                pred, gt,
                classes={1: "pancreas", 2: "tumor"},
                compute_surface_metrics=True,
            ))

    return per_case


def load_model_from_experiment(exp_dir: Path, device: str) -> torch.nn.Module:
    """Load model from experiment directory (reads config + best checkpoint)."""
    cfg_candidates = list(exp_dir.glob("*.yaml")) + [
        Path("configs") / (exp_dir.name.split("_")[0] + ".yaml"),
    ]
    cfg_path = next((p for p in cfg_candidates if p.exists()), None)

    # Fallback: use base config
    if cfg_path is None or not cfg_path.exists():
        cfg_path = Path("configs/supervised_baseline.yaml")

    cfg   = load_config(str(cfg_path))
    model = build_model(cfg["model"]).to(device)

    ckpt_path = exp_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pth found in {exp_dir}")

    load_checkpoint(str(ckpt_path), model, device=device)
    return model, cfg


# ── Main comparison ────────────────────────────────────────────────────────────

def compare_all_methods(
    data_dir:       str,
    experiments_dir: str,
    output_dir:     str,
    device:         str = "cuda",
    val_fold:       int = 0,
):
    device      = torch.device(device if torch.cuda.is_available() else "cpu")
    exp_root    = Path(experiments_dir)
    out_dir     = Path(output_dir)
    fig_dir     = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Discover available experiments
    experiments = {p.name: p for p in sorted(exp_root.iterdir())
                   if p.is_dir() and (p / "best_model.pth").exists()}

    if not experiments:
        print(f"No completed experiments found in {exp_root}")
        print("Expected: experiments/<name>/best_model.pth")
        return

    print(f"Found {len(experiments)} experiments: {list(experiments.keys())}\n")

    # Build validation loader (same for all methods)
    val_ds     = PancreasDataset(data_dir, split="val", val_fold=val_fold)
    val_loader = val_ds.get_loader(batch_size=1, num_workers=4, shuffle=False)
    patch_size = PATCH

    # Evaluate each method
    all_results   = {}
    all_per_case  = {}

    for name, exp_dir in experiments.items():
        print(f"Evaluating: {name}")
        try:
            model, cfg = load_model_from_experiment(exp_dir, str(device))
            num_classes = cfg["model"].get("num_classes", 3)
            per_case    = run_inference(model, val_loader, device,
                                        patch_size, num_classes)
            agg         = aggregate_metrics(per_case)
            all_results[name]  = agg
            all_per_case[name] = per_case
            print(f"  Tumor Dice: {agg.get('tumor_dice_mean', 0):.4f} ± "
                  f"{agg.get('tumor_dice_std', 0):.4f}")
        except Exception as e:
            print(f"  SKIP ({e})")

    if not all_results:
        print("No results to compare.")
        return

    # ── Comparison table ───────────────────────────────────────────────────────
    rows = []
    for method, agg in all_results.items():
        row = {"method": method}
        for metric in METRICS_TO_REPORT:
            mean_k = f"{metric}_mean"
            std_k  = f"{metric}_std"
            if mean_k in agg:
                row[metric] = f"{agg[mean_k]:.3f} ± {agg.get(std_k, 0):.3f}"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("method")
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(df.to_string())
    df.to_csv(out_dir / "comparison_table.csv")
    print(f"\nSaved → {out_dir / 'comparison_table.csv'}")

    # ── Save raw results ───────────────────────────────────────────────────────
    save_results_json(
        {k: v for k, v in all_results.items()},
        str(out_dir / "aggregated_results.json")
    )
    save_results_json(
        {k: v for k, v in all_per_case.items()},
        str(out_dir / "per_case_results.json")
    )

    # ── Figures ────────────────────────────────────────────────────────────────
    _plot_dice_comparison(all_per_case, fig_dir)
    _plot_metric_bars(all_results, fig_dir)
    _plot_labeled_ratio_ablation(all_results, fig_dir)

    print(f"\nAll figures saved to {fig_dir}")


# ── Plotting functions ─────────────────────────────────────────────────────────

def _plot_dice_comparison(all_per_case: dict, fig_dir: Path):
    """Box plot of tumor Dice distribution across methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Segmentation Performance Comparison", fontsize=14, fontweight="bold")

    for ax, cls in zip(axes, ["tumor", "pancreas"]):
        data   = []
        labels = []
        for method, per_case in all_per_case.items():
            values = [r[f"{cls}_dice"] for r in per_case if f"{cls}_dice" in r]
            if values:
                data.append(values)
                labels.append(method.replace("_", "\n"))

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        colors = list(PALETTE.values())[:len(data)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Dice Score")
        ax.set_title(f"{cls.capitalize()} Dice")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="Dice=0.5")

    plt.tight_layout()
    plt.savefig(fig_dir / "dice_comparison_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_metric_bars(all_results: dict, fig_dir: Path):
    """Bar chart comparing mean metrics across methods."""
    metrics    = ["tumor_dice", "pancreas_dice"]
    methods    = list(all_results.keys())
    x          = np.arange(len(methods))
    width      = 0.35

    fig, axes  = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        means  = [all_results[m].get(f"{metric}_mean", 0) for m in methods]
        stds   = [all_results[m].get(f"{metric}_std",  0) for m in methods]
        colors = [PALETTE.get(m, "#607D8B") for m in methods]

        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.8, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=8)
        ax.set_ylabel("Mean Dice ± Std")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1)

        # Annotate bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Mean Segmentation Metrics by Method", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "metric_bars.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_labeled_ratio_ablation(all_results: dict, fig_dir: Path):
    """
    Ablation: tumor Dice vs. labeled data fraction.
    Looks for experiments named like 'semisup_Xpct' or 'semisup'.
    """
    ablation_data = []

    # Supervised baseline = 100%
    for name, agg in all_results.items():
        if "supervised" in name and "semi" not in name:
            ablation_data.append({
                "ratio": 100, "method": "supervised",
                "dice_mean": agg.get("tumor_dice_mean", 0),
                "dice_std":  agg.get("tumor_dice_std",  0),
            })
        elif "semisup" in name:
            # Try to extract ratio from name, else default to 10%
            ratio = 10
            for part in name.split("_"):
                if part.endswith("pct"):
                    try: ratio = int(part[:-3])
                    except: pass
            ablation_data.append({
                "ratio": ratio, "method": name,
                "dice_mean": agg.get("tumor_dice_mean", 0),
                "dice_std":  agg.get("tumor_dice_std",  0),
            })

    if len(ablation_data) < 2:
        return   # not enough data for ablation plot

    ablation_data.sort(key=lambda x: x["ratio"])
    ratios     = [d["ratio"]     for d in ablation_data]
    dice_means = [d["dice_mean"] for d in ablation_data]
    dice_stds  = [d["dice_std"]  for d in ablation_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ratios, dice_means, yerr=dice_stds,
                marker="o", linewidth=2, markersize=8,
                color="#2196F3", capsize=5, label="Semi-supervised")

    # Supervised upper bound
    sup = next((d for d in ablation_data if d["method"] == "supervised"), None)
    if sup:
        ax.axhline(y=sup["dice_mean"], color="#F44336", linestyle="--",
                   linewidth=1.5, label=f"Supervised (100%) = {sup['dice_mean']:.3f}")

    ax.set_xlabel("Labeled Data Fraction (%)")
    ax.set_ylabel("Tumor Dice (mean ± std)")
    ax.set_title("Ablation Study: Effect of Labeled Data Fraction")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_xticks(ratios)

    plt.tight_layout()
    plt.savefig(fig_dir / "ablation_labeled_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare all segmentation methods on the validation set."
    )
    parser.add_argument("--data_dir",       required=True,
                        help="Path to nnUNet Dataset007_Pancreas")
    parser.add_argument("--experiments_dir", default="experiments",
                        help="Root folder containing experiment subdirectories")
    parser.add_argument("--output_dir",      default="results",
                        help="Where to save tables and figures")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--val_fold",        type=int, default=0)
    args = parser.parse_args()

    compare_all_methods(
        data_dir        = args.data_dir,
        experiments_dir = args.experiments_dir,
        output_dir      = args.output_dir,
        device          = args.device,
        val_fold        = args.val_fold,
    )
