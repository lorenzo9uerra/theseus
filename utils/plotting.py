import os

import numpy as np
import wandb
from matplotlib import pyplot as plt

from utils.utils import log


def log_image_to_wandb(out_file, log_key):
    """Logs an image file to WandB, converting SVG/PDF to PNG if necessary."""
    if wandb.run is None:
        return

    if out_file.endswith((".svg", ".pdf")):
        png_file = out_file.replace(".svg", ".png").replace(".pdf", ".png")
        if out_file.endswith(".svg"):
            plt.savefig(png_file, dpi=300, format="png")

        if os.path.exists(png_file):
            wandb.log({log_key: wandb.Image(png_file)})
        else:
            log(f"Skipping WandB log for {out_file} (PNG conversion failed/not found)")
    else:
        wandb.log({log_key: wandb.Image(out_file)})


def plot_anomaly_score_distribution(
    scores, y_truth, out_file, threshold=None, nodes=None, node_to_attack_ids=None
):
    benign_scores = np.array(
        [score for score, label in zip(scores, y_truth, strict=False) if label == 0]
    )

    scores_arr = np.array(scores)
    if np.any(scores_arr > 0):
        min_score = np.min(scores_arr[scores_arr > 0])
        real_max = np.max(scores_arr)
    else:
        min_score = 1e-6
        real_max = 1e-6

    max_score = real_max * 1.5
    if max_score <= min_score:
        max_score = min_score * 10

    bins = np.logspace(np.log10(min_score), np.log10(max_score), 50)
    plt.figure(figsize=(14, 5))

    if len(benign_scores) > 0:
        plt.hist(
            benign_scores,
            bins=bins,
            density=True,
            alpha=0.6,
            label="Benign",
            color="green",
        )

    if nodes is not None and node_to_attack_ids is not None:
        attack_scores = {}
        for score, label, node in zip(scores, y_truth, nodes, strict=False):
            if label != 1:
                continue

            node_key = None
            if node in node_to_attack_ids:
                node_key = node
            elif str(node) in node_to_attack_ids:
                node_key = str(node)
            elif isinstance(node, str) and int(node) in node_to_attack_ids:
                node_key = int(node)

            if node_key is not None:
                for attack_id in node_to_attack_ids[node_key]:
                    attack_scores.setdefault(attack_id, []).append(score)

        tab10 = plt.color_sequences["tab10"]
        attack_colors = [
            tab10[0],
            tab10[1],
            tab10[3],
            tab10[4],
            tab10[5],
            tab10[6],
            tab10[7],
            tab10[8],
            tab10[9],
        ]

        for i, (attack_id, s_list) in enumerate(sorted(attack_scores.items())):
            plt.hist(
                s_list,
                bins=bins,
                density=True,
                alpha=0.6,
                label=f"{attack_id} (n={len(s_list)})",
                color=attack_colors[i % len(attack_colors)],
            )
    else:
        malicious_scores = np.array(
            [score for score, label in zip(scores, y_truth, strict=False) if label == 1]
        )
        if len(malicious_scores) > 0:
            plt.hist(
                malicious_scores,
                bins=bins,
                density=True,
                alpha=0.6,
                label=f"Malicious (n={len(malicious_scores)})",
                color="red",
            )

    if threshold is not None and not np.isinf(threshold):
        plt.axvline(x=threshold, color="black", linestyle="--", linewidth=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Anomaly Score", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    pdf_file = (
        out_file.replace(".png", ".pdf")
        if out_file.endswith(".png")
        else out_file + ".pdf"
    )
    plt.savefig(pdf_file, format="pdf", dpi=300, bbox_inches="tight")

    png_file = (
        out_file.replace(".pdf", ".png") if out_file.endswith(".pdf") else out_file
    )
    plt.savefig(png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    if wandb.run is not None:
        try:
            wandb.log({"anomaly_score_distribution": wandb.Image(png_file)})
        except Exception as e:
            log(f"Warning: Could not log image to wandb: {e}")


def plot_scores_neat(
    scores, y_truth, nodes, node_to_attack_ids, out_file, threshold=None
):
    scores_0 = [
        score
        for i, (score, label) in enumerate(zip(scores, y_truth, strict=False))
        if label == 0 and (score > 0.9 or (score <= 0.9 and i % 500 == 1))
    ]
    scores_1 = [
        (score, node)
        for score, label, node in zip(scores, y_truth, nodes, strict=False)
        if label == 1
    ]

    center_coef = 0.2
    y_zeros = [center_coef] * len(scores_0)
    y_ones = [1 - center_coef] * len(scores_1)

    plt.figure(figsize=(6, 1.5))
    plt.scatter(scores_0, y_zeros, color="green", label="Benign", rasterized=True)

    labels, colors, s_list = [], [], []
    for score, node in scores_1:
        s_list.append(score)
        attack_type = list(node_to_attack_ids.get(node))[0]
        labels.append(f"Attack {attack_type}")
        colors.append("red")

    plt.scatter(s_list, y_ones, color=colors, label=labels, rasterized=True)

    if threshold is not None:
        plt.axvline(
            x=threshold,
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"Threshold: {threshold}",
        )

    plt.xlabel("Node anomaly scores")
    plt.yticks([center_coef, 1 - center_coef], ["Benign", "Malicious"])
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    log_image_to_wandb(out_file, "neat_scores_img_file")
    plt.close()
