import gc
import os
import time
from collections import defaultdict
from datetime import datetime
from time import mktime

import igraph as ig
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import torch
import wandb
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score as ap_score,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

import pidsmaker.utils.labelling as labelling
from pidsmaker.utils.utils import (
    get_all_files_from_folders,
    get_node_to_path_and_type,
    listdir_sorted,
    log,
    mean,
    percentile_90,
    std,
)


def classifier_evaluation(y_test, y_test_pred, scores):
    labels_exist = sum(y_test) > 0
    if labels_exist:
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    else:
        log("WARNING: Computing confusion matrix failed.")
        tn, fp, fn, tp = 1, 1, 1, 1  # only to not break tests

    eps = 1e-12
    fpr = fp / (fp + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    fscore = 2 * (precision * recall) / (precision + recall + eps)

    try:
        auc_val = roc_auc_score(y_test, scores)
    except ValueError as e:
        log(f"WARNING: AUC calculation failed: {e}")
        auc_val = float("nan")
    try:
        ap = ap_score(y_test, scores)
    except ValueError as e:
        log(f"WARNING: AP calculation failed: {e}")
        ap = float("nan")
    try:
        balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    except ValueError as e:
        log(f"WARNING: Balanced ACC calculation failed: {e}")
        balanced_acc = float("nan")

    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    lr_plus = sensitivity / (1 - specificity + eps)
    dor = (tp * tn) / (fp * fn + eps)
    mcc = compute_mcc(tp, fp, tn, fn)

    stats = {
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "fpr": round(fpr, 7),
        "fscore": round(fscore, 5),
        "ap": round(ap, 5),
        "accuracy": round(accuracy, 5),
        "balanced_acc": round(balanced_acc, 5),
        "auc": round(auc_val, 5),
        "lr(+)": round(lr_plus, 5),
        "dor": round(dor, 5),
        "mcc": round(mcc, 5),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    return stats


def compute_mcc(tp, fp, tn, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if denominator == 0:
        return 0

    mcc = numerator / denominator
    return mcc


def get_threshold(val_tw_path, threshold_method: str):
    threshold_method = threshold_method.strip()
    if threshold_method == "max_val_loss":
        return calculate_threshold(val_tw_path, threshold_method)["max"]
    elif threshold_method == "mean_val_loss":
        return calculate_threshold(val_tw_path, threshold_method)["mean"]
    elif threshold_method == "threatrace":
        return 1.5
    elif threshold_method == "flash":
        return 0.53
    elif threshold_method == "nodlink":
        return calculate_threshold(val_tw_path, threshold_method)["percentile_90"]
    elif threshold_method == "magic":
        return calculate_threshold(val_tw_path, threshold_method)["mean"]
    elif threshold_method == "best_mcc":
        # This requires ground truth, so return None here
        # The actual threshold will be computed in get_best_mcc_threshold()
        return None
    raise ValueError(f"Invalid threshold method `{threshold_method}`")


def get_best_mcc_threshold(val_tw_path, ground_truth_nids, cfg, use_dst_node_loss=True):
    """
    Find the optimal threshold that maximizes MCC on the validation set.
    """
    from sklearn.metrics import matthews_corrcoef

    node_to_info = get_node_to_path_and_type(cfg)
    node_to_losses = defaultdict(list)

    filelist = listdir_sorted(val_tw_path)
    for file in sorted(filelist):
        filepath = os.path.join(val_tw_path, file)
        df = pd.read_csv(filepath).to_dict(orient="records")

        # Check if this is node-level or edge-level format
        if "node" in df[0]:
            # Node-level format
            for line in df:
                node = line["node"]
                loss = line["loss"]
                node_to_losses[node].append(loss)
        else:
            # Edge-level format
            for line in df:
                srcnode = line["srcnode"]
                dstnode = line["dstnode"]
                loss = line["loss"]

                node_to_losses[srcnode].append(loss)
                if use_dst_node_loss:
                    node_to_losses[dstnode].append(loss)

    node_scores = {}
    num_filtered = 0
    for node_id, losses in node_to_losses.items():
        if node_id not in node_to_info or node_to_info[node_id]["type"] != "subject":
            num_filtered += 1
            continue
        node_scores[node_id] = max(losses)

    log(
        f"MCC threshold optimization: filtered {num_filtered} non-process nodes, using {len(node_scores)} process nodes"
    )

    nodes = list(node_scores.keys())
    scores = np.array([node_scores[n] for n in nodes])
    y_true = np.array([int(n in ground_truth_nids) for n in nodes])

    # If no positive samples in validation, fall back to max threshold
    if y_true.sum() == 0:
        log(
            "WARNING: No positive samples in validation set for MCC optimization, using max threshold"
        )
        return max(scores), 0.0

    unique_scores = np.unique(scores)

    if len(unique_scores) > 1000:
        percentiles = np.percentile(scores, np.linspace(0, 100, 1000))
        candidate_thresholds = np.unique(percentiles)
    else:
        candidate_thresholds = unique_scores

    best_mcc = -2  # MCC ranges from -1 to 1
    best_threshold = candidate_thresholds[0]

    for thr in candidate_thresholds:
        y_pred = (scores > thr).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thr

    log(f"Best MCC threshold optimization: threshold={best_threshold:.5f}, mcc={best_mcc:.4f}")

    y_pred_best = (scores > best_threshold).astype(int)
    tp = ((y_pred_best == 1) & (y_true == 1)).sum()
    fp = ((y_pred_best == 1) & (y_true == 0)).sum()
    fn = ((y_pred_best == 0) & (y_true == 1)).sum()
    tn = ((y_pred_best == 0) & (y_true == 0)).sum()
    log(f"  Validation stats at best threshold: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    return float(best_threshold), float(best_mcc)


def reduce_losses_to_score(losses: list[float], threshold_method: str):
    threshold_method = threshold_method.strip()
    if threshold_method == "mean_val_loss":
        return np.mean(losses)
    elif (
        threshold_method == "max_val_loss"
        or threshold_method == "threatrace"
        or threshold_method == "flash"
        or threshold_method == "nodlink"
        or threshold_method == "best_mcc"
    ):
        return np.max(losses)
    raise ValueError(f"Invalid threshold method {threshold_method}")


def calculate_threshold(val_tw_dir, threshold_method):
    filelist = listdir_sorted(val_tw_dir)

    loss_list = []
    for file in sorted(filelist):
        f = os.path.join(val_tw_dir, file)
        df = pd.read_csv(f).to_dict()
        if threshold_method == "magic":
            loss_list.extend(df["magic_score"].values())
        else:
            loss_list.extend(df["loss"].values())

    thr = {
        "max": max(loss_list),
        "mean": mean(loss_list),
        "percentile_90": percentile_90(loss_list),
    }
    log(
        f"Thresholds: MEAN={thr['mean']:.3f}, STD={std(loss_list):.3f}, MAX={thr['max']:.3f}, 90 Percentile={thr['percentile_90']:.3f}"
    )

    return thr


def plot_precision_recall(scores, y_truth, out_file):
    precision, recall, thresholds = precision_recall_curve(y_truth, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    precision_ticks = [i / 20 for i in range(7)]  # Generates [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    plt.yticks(precision_ticks)

    plt.savefig(out_file)


def plot_simple_scores(scores, y_truth, out_file):
    scores_0 = [score for score, label in zip(scores, y_truth) if label == 0]
    scores_1 = [score for score, label in zip(scores, y_truth) if label == 1]

    y_zeros = [0] * len(scores_0)
    y_ones = [1] * len(scores_1)

    plt.figure(figsize=(6, 2))
    plt.scatter(scores_0, y_zeros, color="green")
    plt.scatter(scores_1, y_ones, color="red")

    plt.xlabel("Node anomaly scores")
    plt.yticks([0, 1], ["Benign", "Malicious"])
    plt.ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(out_file)


def plot_score_seen(scores, y_truth, out_file):
    scores_0 = [score for score, label in zip(scores, y_truth) if label == 0]
    scores_1 = [score for score, label in zip(scores, y_truth) if label == 1]

    y_zeros = [0] * len(scores_0)
    y_ones = [1] * len(scores_1)

    plt.figure(figsize=(6, 2))
    plt.scatter(scores_0, y_zeros, color="green")
    plt.scatter(scores_1, y_ones, color="red")

    plt.xlabel("Node anomaly scores")
    plt.yticks([0, 1], ["Seen", "Unseen"])
    plt.ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(out_file)


def plot_anomaly_score_distribution(
    scores, y_truth, out_file, threshold=None, nodes=None, node_to_attack_ids=None
):
    """
    Plot histogram distribution of anomaly scores for benign vs malicious nodes.
    Uses log scale on both axes for better visualization of score distributions.
    Saves both PDF and PNG versions for compatibility with WandB.
    """
    benign_scores = np.array([score for score, label in zip(scores, y_truth) if label == 0])

    scores_arr = np.array(scores)
    min_score = np.min(scores_arr[scores_arr > 0]) if np.any(scores_arr > 0) else 1e-6
    max_score = np.max(scores_arr)
    if max_score <= min_score:
        max_score = min_score + 1.0

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
        for score, label, node in zip(scores, y_truth, nodes):
            if label == 1:
                node_key = None
                if node in node_to_attack_ids:
                    node_key = node
                elif str(node) in node_to_attack_ids:
                    node_key = str(node)
                elif isinstance(node, str) and int(node) in node_to_attack_ids:
                    node_key = int(node)

                if node_key is not None:
                    for attack_id in node_to_attack_ids[node_key]:
                        if attack_id not in attack_scores:
                            attack_scores[attack_id] = []
                        attack_scores[attack_id].append(score)

        tab10_colors = plt.color_sequences["tab10"]
        attack_colors = [
            tab10_colors[0],
            tab10_colors[1],
            tab10_colors[3],
            tab10_colors[4],
            tab10_colors[5],
            tab10_colors[6],
            tab10_colors[7],
            tab10_colors[8],
            tab10_colors[9],
        ]

        for i, (attack_id, attack_score_list) in enumerate(sorted(attack_scores.items())):
            color = attack_colors[i % len(attack_colors)]
            plt.hist(
                attack_score_list,
                bins=bins,
                density=True,
                alpha=0.6,
                label=f"{attack_id} (n={len(attack_score_list)})",
                color=color,
            )
    else:
        malicious_scores = np.array([score for score, label in zip(scores, y_truth) if label == 1])
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
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    pdf_file = out_file.replace(".png", ".pdf") if out_file.endswith(".png") else out_file + ".pdf"
    plt.savefig(pdf_file, format="pdf", dpi=300, bbox_inches="tight")

    png_file = out_file.replace(".pdf", ".png") if out_file.endswith(".pdf") else out_file
    plt.savefig(png_file, format="png", dpi=300, bbox_inches="tight")

    plt.close()

    if wandb.run is not None:
        try:
            wandb.log({"anomaly_score_distribution": wandb.Image(png_file)})
        except Exception as e:
            log(f"Warning: Could not log image to wandb: {e}")


def plot_scores_with_paths_node_level(
    scores,
    y_truth,
    nodes,
    max_val_loss_tw,
    tw_to_malicious_nodes,
    node2attacks,
    out_file,
    cfg,
    threshold=None,
):
    node_to_path = get_node_to_path_and_type(cfg)
    paths, types = [], []
    for n in nodes:
        types.append(node_to_path[n]["type"])
        path = node_to_path[n]["path"]
        if path == "None":
            paths.append(node_to_path[n]["cmd"] if "cmd" in node_to_path[n] else path)
        else:
            paths.append(path)

    scores = np.array(scores)
    y_truth = np.array(y_truth)
    types = np.array(types)
    nodes = np.array(nodes)

    # Define marker styles for each type
    marker_styles = {
        "subject": "s",  # Square
        "file": "o",  # Circle
        "netflow": "D",  # Diamond
    }

    # Separate the scores based on labels
    scores_0 = scores[y_truth == 0]
    scores_1 = scores[y_truth == 1]
    types_0 = types[y_truth == 0]
    types_1 = types[y_truth == 1]
    paths_0 = [path for path, label in zip(paths, y_truth) if label == 0]
    paths_1 = [path for path, label in zip(paths, y_truth) if label == 1]

    plt.figure(figsize=(12, 6))

    red = (155 / 255, 44 / 255, 37 / 255)
    green = (62 / 255, 126 / 255, 42 / 255)

    attack_colors = {
        0: "black",
        1: "red",
        2: "blue",
    }

    node2attack = np.array(
        [
            list(node2attacks.get(node, {0}))[0] if node2attacks.get(node) else 0
            for node in nodes[y_truth == 1]
        ]
    )

    # Plot each type with a different marker for Label 0
    for t in marker_styles.keys():
        plt.scatter(
            scores_0[types_0 == t],
            [0] * sum(types_0 == t),
            marker=marker_styles[t],
            color=green,
            label=t,
        )

    # Plot each type with a different marker for Label 1
    for t in marker_styles.keys():
        plt.scatter(
            scores_1[types_1 == t],
            [1] * sum(types_1 == t),
            marker=marker_styles[t],
            color=[attack_colors.get(c, "gray") for c in node2attack[types_1 == t]],
        )

    # Adding labels and title
    plt.xlabel("Scores")
    plt.ylabel("Labels")
    plt.yticks([0, 1], ["0", "1"])  # Set y-ticks to show label categories
    plt.title("Scatter Plot of Scores by Label")
    plt.legend()

    # Add vertical line at threshold if provided
    if threshold is not None:
        plt.axvline(x=threshold, color="purple", linestyle="--", label="Threshold")
        plt.legend()

    combined_scores = list(zip(scores, paths, y_truth, max_val_loss_tw, nodes))

    # Sort combined list by scores in descending order
    combined_scores_sorted = sorted(combined_scores, key=lambda x: x[0], reverse=True)

    # Separate the top scores by their labels
    keep_only = 14
    top_0 = [item for item in combined_scores_sorted if item[2] == 0][:keep_only]
    top_1 = [item for item in combined_scores_sorted if item[2] == 1][:keep_only]

    x_axis = max(scores) + max(scores) * 0.05
    # Annotate the top scores for label 0
    for i, (score, path, _, max_tw_idx, node) in enumerate(top_0):
        y_position = 0 - (i * 0.1)  # Adjust y-position for each label to avoid overlap
        plt.text(
            x_axis,
            y_position,
            f"{str(path)[:30]} ({score:.2f}): TW {max_tw_idx}, Node: {node}",
            fontsize=8,
            va="center",
            ha="left",
            color=green,
        )

    # Annotate the top scores for label 1
    for i, (score, path, _, max_tw_idx, node) in enumerate(top_1):
        y_position = 1.4 - (
            i * 0.1
        )  # Adjust y-position for each label to avoid overlap and add space between groups
        plt.text(
            x_axis,
            y_position,
            f"{str(path)[:30]} ({score:.2f}): TW {max_tw_idx}, Node: {node}",
            fontsize=8,
            va="center",
            ha="left",
            color=red,
        )

    plt.text(
        min(scores),
        1.6,
        f"Dataset: {cfg.dataset.name}",
        fontsize=8,
        va="center",
        ha="left",
        color="black",
    )
    plt.text(
        min(scores),
        1.5,
        f"Malicious TW: {str(list(tw_to_malicious_nodes.keys()))}",
        fontsize=8,
        va="center",
        ha="left",
        color="black",
    )

    plt.xlim([min(scores), max(scores) * 1.5])  # Adjust xlim to make space for text
    plt.ylim([-1, 2])  # Adjust ylim to ensure the text is within the figure bounds
    plt.savefig(out_file)


def plot_scores_with_paths_edge_level(
    scores, y_truth, edges, tw_to_malicious_nodes, node2attacks, out_file, cfg, threshold=None
):
    node_to_path = get_node_to_path_and_type(cfg)
    paths, types = [], []
    # Prints the path if it exists, else tries to print the cmd line
    for src, dst, *_ in edges:
        src = int(src)
        dst = int(dst)
        src_type = node_to_path[src]["type"]
        dst_type = node_to_path[dst]["type"]
        types.append(src_type)

        path_src = (
            node_to_path[src]["path"] + ", " + node_to_path[src]["cmd"]
            if src_type == "subject"
            else node_to_path[src]["path"]
        )
        path_dst = (
            node_to_path[dst]["path"] + ", " + node_to_path[dst]["cmd"]
            if dst_type == "subject"
            else node_to_path[dst]["path"]
        )
        paths.append((path_src, path_dst))

    # Convert data to numpy arrays for easy manipulation
    scores = np.array(scores)
    y_truth = np.array(y_truth)
    types = np.array(types)

    # Define marker styles for each type
    marker_styles = {
        "subject": "s",  # Square
        "file": "o",  # Circle
        "netflow": "D",  # Diamond
    }

    # Separate the scores based on labels
    scores_0 = scores[y_truth == 0]
    scores_1 = scores[y_truth == 1]
    types_0 = types[y_truth == 0]
    types_1 = types[y_truth == 1]

    plt.figure(figsize=(14, 6))

    red = (155 / 255, 44 / 255, 37 / 255)
    green = (62 / 255, 126 / 255, 42 / 255)

    attack_colors = {
        0: "black",
        1: "red",
        2: "blue",
    }

    malicious_elements = [n for y_true, n in zip(y_truth, edges) if y_true == 1]
    node2attack = np.array(
        [
            list(node2attacks.get(node, {0}))[0] if node2attacks.get(node) else 0
            for node in malicious_elements
        ]
    )

    # Plot each type with a different marker for Label 0
    for t in marker_styles.keys():
        plt.scatter(
            scores_0[types_0 == t],
            [0] * sum(types_0 == t),
            marker=marker_styles[t],
            color=green,
            label=t,
        )

    # Plot each type with a different marker for Label 1
    for t in marker_styles.keys():
        plt.scatter(
            scores_1[types_1 == t],
            [1] * sum(types_1 == t),
            marker=marker_styles[t],
            color=[attack_colors.get(c, "gray") for c in node2attack[types_1 == t]],
        )

    # Adding labels and title
    plt.xlabel("Scores")
    plt.ylabel("Labels")
    plt.yticks([0, 1], ["0", "1"])  # Set y-ticks to show label categories
    plt.title("Scatter Plot of Scores by Label")
    plt.legend()

    # Add vertical line at threshold if provided
    if threshold is not None:
        plt.axvline(x=threshold, color="purple", linestyle="--", label="Threshold")
        plt.legend()

    # Combine scores and paths for easy handling
    combined_scores = list(zip(scores, paths, y_truth, edges))

    # Sort combined list by scores in descending order
    combined_scores_sorted = sorted(combined_scores, key=lambda x: x[0], reverse=True)

    # Separate the top scores by their labels
    keep_only = 14
    top_0 = [item for item in combined_scores_sorted if item[2] == 0][:keep_only]
    top_1 = [item for item in combined_scores_sorted if item[2] == 1][:keep_only]

    x_axis = max(scores) + max(scores) * 0.02
    # Annotate the top scores for label 0
    for i, (score, path, _, (src, dst, _, edge_type)) in enumerate(top_0):
        y_position = 0 - (i * 0.1)  # Adjust y-position for each label to avoid overlap
        edge_type = edge_type.replace("EVENT_", "")
        plt.text(
            x_axis,
            y_position,
            f"({src}, {dst} | {node_to_path[int(src)]['type']}, {edge_type}, {node_to_path[int(dst)]['type']}): {str(path[0])[:28]} => {str(path[1])[:28]} ({score:.2f})",
            fontsize=6,
            va="center",
            ha="left",
            color=green,
        )

    # Annotate the top scores for label 1
    for i, (score, path, _, (src, dst, _, edge_type)) in enumerate(top_1):
        y_position = 1.4 - (
            i * 0.1
        )  # Adjust y-position for each label to avoid overlap and add space between groups
        edge_type = edge_type.replace("EVENT_", "")
        plt.text(
            x_axis,
            y_position,
            f"({src}, {dst} | {node_to_path[int(src)]['type']}, {edge_type}, {node_to_path[int(dst)]['type']}): {str(path[0])[:28]} => {str(path[1])[:28]} ({score:.2f})",
            fontsize=6,
            va="center",
            ha="left",
            color=red,
        )
    plt.text(
        min(scores),
        1.6,
        f"Dataset: {cfg.dataset.name}",
        fontsize=8,
        va="center",
        ha="left",
        color="black",
    )
    plt.text(
        min(scores),
        1.5,
        f"Malicious TW: {str(list(tw_to_malicious_nodes.keys()))}",
        fontsize=8,
        va="center",
        ha="left",
        color="black",
    )

    plt.xlim([min(scores), max(scores) * 1.5])  # Adjust xlim to make space for text
    plt.ylim([-1, 2])  # Adjust ylim to ensure the text is within the figure bounds
    plt.savefig(out_file)


def plot_scores_neat(scores, y_truth, nodes, node2attacks, out_file, threshold=None):
    # Separate scores based on labels
    scores_0 = [
        score
        for i, (score, label) in enumerate(zip(scores, y_truth))
        if label == 0 and (score > 0.9 or (score <= 0.9 and i % 500 == 1))
    ]
    scores_1 = [(score, node) for score, label, node in zip(scores, y_truth, nodes) if label == 1]

    # Assign different colors for each attack type
    attack_colors = {
        0: "black",
        1: "red",
        2: "blue",
    }

    center_coef = 0.2  # to center the lines/dots

    # Positions on the y-axis for the scatter plot
    y_zeros = [center_coef] * len(scores_0)  # All zeros at y=0
    y_ones = [1 - center_coef] * len(scores_1)  # All ones at y=1

    plt.figure(figsize=(6, 1.5))  # Width, height in inches
    plt.scatter(scores_0, y_zeros, color="green", label="Benign", rasterized=True)

    # Plot each malicious node with its corresponding color based on its attack type
    labels, colors, scores = [], [], []
    for score, node in scores_1:
        scores.append(score)
        attack_type = list(node2attacks.get(node, {0}))[0] if node2attacks.get(node) else 0
        labels.append(f"Attack {attack_type}")
        colors.append(attack_colors.get(attack_type, "gray"))
    plt.scatter(scores, y_ones, color=colors, label=labels, rasterized=True)

    if threshold is not None:
        plt.axvline(
            x=threshold, color="black", linestyle="-", linewidth=2, label=f"Threshold: {threshold}"
        )

    plt.xlabel("Node anomaly scores")
    plt.yticks([center_coef, 1 - center_coef], ["Benign", "Malicious"])
    plt.ylim(-0.1, 1.1)  # Adjust if necessary to bring them even closer

    plt.tight_layout()  # Ensures everything fits within the figure area
    plt.savefig(out_file, dpi=300)


def plot_false_positives(y_true, y_pred, out_file):
    plt.figure(figsize=(10, 6))

    plt.plot(y_pred, label="y_pred", color="blue")

    # Adding green dots for true positives (y_true == 1)
    label_indices = [i for i, true in enumerate(y_true) if true == 1]
    plt.scatter(
        label_indices, [y_pred[i] for i in label_indices], color="green", label="True Positive"
    )

    # Adding red dots for false positives (y_true == 0 and y_pred == 1)
    false_positive_indices = [
        i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true == 0 and pred == 1
    ]
    plt.scatter(
        false_positive_indices,
        [y_pred[i] for i in false_positive_indices],
        color="red",
        label="False Positive",
    )

    plt.xlabel("Index")
    plt.ylabel("Prediction Value")
    plt.title("True Positives and False Positives in Predictions")
    plt.legend()
    plt.savefig(out_file)


def plot_dor_recall_curve(scores, y_truth, out_file):
    scores = np.array(scores)
    y_truth = np.array(y_truth)
    thresholds = np.linspace(scores.min(), scores.max(), 300)

    sensitivity_list = []
    dor_list = []

    # Iterate over each threshold to calculate recall and DOR
    for threshold in thresholds:
        # Make predictions based on the threshold
        predictions = scores >= threshold

        # Calculate TP, FP, TN, FN
        TN, FP, FN, TP = confusion_matrix(y_truth, predictions).ravel()
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Calculate Diagnostic Odds Ratio (DOR)
        if (FP * FN) == 0:
            dor = np.nan
        else:
            dor = (TP * TN) / (FP * FN)

        sensitivity_list.append(recall)
        dor_list.append(dor)

    # Convert lists to numpy arrays for plotting
    sensitivity_list = np.array(sensitivity_list)
    dor_list = np.array(dor_list)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(sensitivity_list, dor_list, label="DOR vs Sensitivity", color="blue", marker="o")
    plt.xlabel("Sensitivity")
    plt.ylabel("Diagnostic Odds Ratio (DOR)")
    plt.title("Diagnostic Odds Ratio vs Sensitivity at Different Thresholds")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_file)


def plot_detected_attacks_vs_precision(scores, nodes, node2attacks, labels, out_file):
    """
    Computes the Attack Detection Precision (ADP) curve and score.
    ADP measures the % of unique attacks detected vs. Precision across thresholds.
    """
    scores = np.array(scores)
    labels = np.array(labels)
    nodes = np.array(nodes)

    # For large datasets, subsample to avoid OOM
    n_samples = len(scores)
    max_samples = 500_000

    if n_samples > max_samples:
        # Keep all positive samples + subsample negatives
        pos_mask = labels == 1
        neg_indices = np.where(~pos_mask)[0]

        n_neg_keep = max_samples - pos_mask.sum()
        if n_neg_keep > 0 and len(neg_indices) > n_neg_keep:
            np.random.seed(42)  # For reproducibility
            neg_indices = np.random.choice(neg_indices, size=int(n_neg_keep), replace=False)

        pos_indices = np.where(pos_mask)[0]
        keep_indices = np.concatenate([pos_indices, neg_indices])

        scores = scores[keep_indices]
        labels = labels[keep_indices]
        nodes = nodes[keep_indices]
        log(f"Subsampled from {n_samples:,} to {len(scores):,} nodes for ADP calculation")

    # Sort nodes by descending anomaly scores (strict to lenient threshold)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_nodes = nodes[sorted_indices]  # Keep as numpy array
    sorted_labels = labels[sorted_indices]  # Keep as numpy array

    # Get total number of unique attacks in the evaluation set
    # Only count attacks that have nodes present in this evaluation
    attacks_in_eval_set = set()
    for node in nodes:
        node = int(node)
        if node in node2attacks:
            attacks_in_eval_set.update(node2attacks[node])

    total_attacks = len(attacks_in_eval_set)
    if total_attacks == 0:
        log("Warning: No attacks found in this evaluation set")
        return 0.0

    log(f"ADP: Found {total_attacks} attack(s) in evaluation set: {sorted(attacks_in_eval_set)}")

    n_nodes = len(sorted_nodes)
    max_curve_points = 10_000  # Limit curve points to avoid OOM

    if n_nodes > max_curve_points:
        # Sample indices at regular intervals, always including first and last
        sample_indices = np.linspace(0, n_nodes - 1, max_curve_points, dtype=int)
        sample_indices = np.unique(sample_indices)  # Remove duplicates
    else:
        sample_indices = np.arange(n_nodes)

    detected_attacks = set()
    precisions = []
    detected_percentages = []

    tp = 0  # True positives
    fp = 0  # False positives
    next_sample_idx = 0

    # Sweep through thresholds by processing nodes in descending score order
    for i, (node, label) in enumerate(zip(sorted_nodes, sorted_labels)):
        node = int(node)
        # Update counts based on ground truth label
        if label == 1:
            tp += 1
            # Track which attacks we've detected
            if node in node2attacks:
                detected_attacks.update(node2attacks[node])
        else:
            fp += 1

        # Only record curve point at sampled indices
        if next_sample_idx < len(sample_indices) and i == sample_indices[next_sample_idx]:
            if tp + fp > 0:
                precision = tp / (tp + fp)
                detected_percentage = (len(detected_attacks) / total_attacks) * 100
                precisions.append(precision)
                detected_percentages.append(detected_percentage)
            next_sample_idx += 1

    # Free memory
    del sorted_nodes, sorted_labels, sorted_indices
    gc.collect()

    # If we have no valid points, return 0
    if len(precisions) == 0:
        log("Warning: No valid threshold points found for ADP")
        return 0.0

    # Sort by precision (ascending) for correct trapezoidal integration
    sorted_pairs = sorted(zip(precisions, detected_percentages))
    precisions_sorted = np.array([p for p, _ in sorted_pairs])
    detected_sorted = np.array([d for _, d in sorted_pairs])

    # Free memory
    del precisions, detected_percentages
    gc.collect()

    # Calculate area under curve using trapezoidal rule
    # Normalize by dividing by 100 since detected_percentages is in [0, 100]
    area_under_curve = np.trapz(detected_sorted, precisions_sorted) / 100.0

    # Plotting
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(
            precisions_sorted,
            detected_sorted,
            color="b",
            linewidth=2,
            label=f"ADP Score = {area_under_curve:.3f}",
        )
        plt.fill_between(precisions_sorted, detected_sorted, color="blue", alpha=0.2)
        plt.xlabel("Precision (High = Few False Positives)", fontsize=12)
        plt.ylabel("% of Detected Attacks", fontsize=12)
        plt.title("Attack Detection vs Precision", fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.xlim(0, 1)
        plt.ylim(0, 100.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        log(f"Error while generating ADP plot: {e}")
        plt.close()

    return area_under_curve


def plot_recall_vs_precision(scores, nodes, node2attacks, labels, out_file):
    """
    Plot the recall vs precision using anomaly scores, node IDs, ground truth labels,
    and a mapping of node IDs to their respective attack numbers.

    This function calculates precision on the x-axis and recall on the y-axis, handles duplicate
    x-values by taking the maximum y-value for each unique x-value, and then plots it with a
    filled area under the curve.
    Ensures that the plot starts at (0, 0).
    """
    # Sort nodes by descending anomaly scores
    sorted_indices = np.argsort(scores)[::-1]
    sorted_nodes = [nodes[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # Initialize variables for tracking true positives and recall
    total_positives = np.sum(labels)  # Total number of actual positives
    tp = 0  # True positives
    fp = 0  # False positives

    recalls = [0]  # Start at y=0
    precisions = [0]  # Start at x=0

    # Count true positives and calculate precision and recall at each threshold
    for i, node in enumerate(sorted_nodes):
        # Update tp and fp based on the current label
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1

        # Calculate precision and recall
        precision = tp / (tp + fp)
        recall = tp / total_positives

        precisions.append(precision)
        recalls.append(recall)

    # Use max recall for each unique precision
    precision_to_recall = defaultdict(list)
    for precision, recall in zip(precisions, recalls):
        precision_to_recall[precision].append(recall)

    unique_precisions = []
    max_recalls = []
    for precision, recall_list in sorted(precision_to_recall.items()):
        unique_precisions.append(precision)
        max_recalls.append(np.max(recall_list))

    # Calculate area under the curve for recall vs precision
    area_under_curve = np.trapz(max_recalls, unique_precisions) / 100

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        unique_precisions,
        max_recalls,
        color="b",
        label=f"Area under curve = {area_under_curve:.2f}",
    )
    plt.fill_between(unique_precisions, max_recalls, color="blue", alpha=0.2)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Recall vs Precision")
    plt.legend(loc="lower left")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(out_file)
    return area_under_curve


def plot_discrimination_metric(scores, y_truth, out_file):
    scores = np.array(scores)
    y_truth = np.array(y_truth)
    anomalous_scores = scores[y_truth == 1]
    benign_scores = scores[y_truth == 0]

    # Define the top K value
    K = len(anomalous_scores)
    if K == 0:
        return 0.0

    # Get top K scores for anomalies and benign samples
    top_anomalous_scores = np.sort(anomalous_scores)[-K:][::-1]
    top_benign_scores = np.sort(benign_scores)[-K:][::-1]

    # Calculate the boolean mask where anomalies > benign
    mask = top_anomalous_scores > top_benign_scores

    # Compute the raw separation area
    if mask.sum() > 0:
        # Positive overlap exists
        raw_area = np.trapz(top_anomalous_scores[mask] - top_benign_scores[mask], dx=1)
    else:
        # No positive overlap; compute negative area
        raw_area = -np.trapz(top_benign_scores - top_anomalous_scores, dx=1)

    # Calculate the range of scores (y-axis normalization)
    max_score = max(top_anomalous_scores.max(), top_benign_scores.max())
    min_score = min(top_anomalous_scores.min(), top_benign_scores.min())
    score_range = max_score - min_score

    # Normalize the area
    total_plot_area = score_range * K
    area = raw_area / total_plot_area

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.plot(range(K), top_anomalous_scores, label="Top Anomalies", color="red")
    plt.plot(range(K), top_benign_scores, label="Top Benign", color="blue")

    # Highlight the positive or negative area
    if mask.sum() > 0:
        # Positive overlap exists
        plt.fill_between(
            range(K),
            top_anomalous_scores,
            top_benign_scores,
            where=(top_anomalous_scores > top_benign_scores),
            interpolate=True,
            color="purple",
            alpha=0.2,
            label=f"Positive Area {area:.2f}",
        )
    else:
        # No positive overlap; negative area
        plt.fill_between(
            range(K),
            top_anomalous_scores,
            top_benign_scores,
            where=(top_anomalous_scores < top_benign_scores),
            interpolate=True,
            color="orange",
            alpha=0.2,
            label=f"Negative Area {area:.2f}",
        )

    plt.xlabel("Rank")
    plt.ylabel("Anomaly Score")
    plt.title("Top K Anomalous vs Benign Scores")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(out_file)
    return area


def compute_discrimination_score(pred_scores, nodes, node2attacks, y_truth, k=10):
    pred_scores = np.array(pred_scores).astype(float)
    y_truth = np.array(y_truth)

    # Handle edge case: no attacks in this evaluation set
    if not node2attacks:
        log("Warning: No attacks found for discrimination score computation")
        return {"discrimination": 0.0}

    pred_scores /= pred_scores.max() + 1e-6
    attack2max_score = defaultdict(float)

    for node, score in zip(nodes, pred_scores):
        if node in node2attacks:
            for attack in node2attacks[node]:
                attack2max_score[attack] = max(attack2max_score[attack], score)
    attack2max_score = dict(sorted(attack2max_score.items(), key=lambda item: item[0]))

    benign_scores = pred_scores[y_truth == 0]

    # For large datasets, use partition instead of full sort for efficiency
    if len(benign_scores) > k:
        # Use argpartition to find top k efficiently without full sort
        top_k_indices = np.argpartition(benign_scores, -k)[-k:]
        top_benign_scores = benign_scores[top_k_indices]
        top_benign_scores = np.sort(top_benign_scores)[::-1]  # Sort only the top k
    else:
        top_benign_scores = np.sort(benign_scores)[::-1]

    mean = np.mean(top_benign_scores) if len(top_benign_scores) > 0 else 0.0
    att2score = {}

    # Get all unique attack IDs from node2attacks
    all_attacks = set.union(*list(node2attacks.values())) if node2attacks else set()
    for att in all_attacks:
        att2score[f"discrim_score_att_{att}"] = 0

    for k, v in attack2max_score.items():
        score = v - mean
        att2score[f"discrim_score_att_{k}"] = score
    att2score["discrimination"] = np.mean(list(att2score.values())) if att2score else 0.0

    # Free memory
    del benign_scores, top_benign_scores
    gc.collect()

    return att2score


def compute_discrimination_tp(pred_scores, nodes, node2attacks, y_truth, k=10):
    pred_scores = np.array(pred_scores).astype(float)
    y_truth = np.array(y_truth)

    # Handle edge case: no attacks in this evaluation set
    if not node2attacks:
        log("Warning: No attacks found for discrimination TP computation")
        return {"discrim_tp_att_sum": 0}

    pred_scores /= pred_scores.max() + 1e-6

    benign_scores = pred_scores[y_truth == 0]

    # For large datasets, use partition instead of full sort
    if len(benign_scores) > k:
        top_k_indices = np.argpartition(benign_scores, -k)[-k:]
        top_benign_scores = benign_scores[top_k_indices]
        top_benign_scores = np.sort(top_benign_scores)[::-1]
    else:
        top_benign_scores = np.sort(benign_scores)[::-1]

    mean = np.mean(top_benign_scores) if len(top_benign_scores) > 0 else 0.0
    att2tp = defaultdict(int)

    # Get all unique attack IDs from node2attacks
    all_attacks = set.union(*list(node2attacks.values())) if node2attacks else set()
    for att in all_attacks:
        att2tp[f"discrim_tp_att_{att}"] = 0

    for node, score in zip(nodes, pred_scores):
        if node in node2attacks:
            for attack in node2attacks[node]:
                if score > mean:
                    att2tp[f"discrim_tp_att_{attack}"] += 1
    att2tp["discrim_tp_att_sum"] = np.sum(list(att2tp.values()))

    # Free memory
    del benign_scores, top_benign_scores
    gc.collect()

    return att2tp


def get_detected_tps(scores, src_dst_t_type, edge2attack, y_truth, cfg):
    """
    Maps each attack to edges that are true positives based on scores.

    Args:
        scores (np.ndarray or list): Confidence scores for each edge.
        src_dst_t_type (list of tuples): List of (src, dst, t, type) tuples representing edges.
        edge2attack (dict): Dictionary mapping edge indices to attack identifiers.
        y_truth (np.ndarray or list): Ground truth labels (1 for true, 0 for false).

    Returns:
        dict: Dictionary mapping attack identifiers to lists of detected true positive edges (as tuples).
    """
    # Ensure inputs are numpy arrays for easier manipulation
    scores = np.array(scores)
    y_truth = np.array(y_truth)

    # Create an array of indices to sort by scores
    indices = np.argsort(scores)[::-1]  # Sort in descending order

    # Sort scores, edges, and labels based on descending scores
    sorted_scores = scores[indices]
    sorted_edges = [src_dst_t_type[i] for i in indices]
    sorted_labels = y_truth[indices]

    # Find the cutoff k where no false positives (label 0) are included
    k = 0
    for i, label in enumerate(sorted_labels):
        if label == 0:  # Stop at the first false positive
            k = i
            break
        k = i + 1  # Include all true positives up to this point

    # Select the top k edges (true positives)
    true_positive_edges = sorted_edges[:k]
    true_positive_indices = indices[:k]

    node_to_path = get_node_to_path_and_type(cfg)

    get_label = (
        lambda data: f"{data.get('path')}-{data.get('cmd')}".replace("None-", "")
        if data["type"] == "subject"
        else data.get("path")
    )

    # Map true positive edges to their attacks
    attack_to_detected_edges = defaultdict(list)
    for edge in true_positive_edges:
        attack = list(edge2attack.get(edge))[0]
        if attack is not None:  # Only include edges with a valid attack mapping
            src_node = node_to_path[int(edge[0])]
            dst_node = node_to_path[int(edge[1])]
            edge = (
                get_label(src_node),
                get_label(dst_node),
                src_node["type"],
                dst_node["type"],
            ) + edge
            attack_to_detected_edges[attack].append(edge)

    return dict(attack_to_detected_edges)


def get_detected_tps_node_level(scores, nodes, node2attack, y_truth, cfg):
    """
    Maps each attack to nodes that are true positives based on scores.

    Args:
        scores (np.ndarray or list): Confidence scores for each node.
        nodes (list of tuples): List of nodes.
        node2attack (dict): Dictionary mapping node indices to attack identifiers.
        y_truth (np.ndarray or list): Ground truth labels (1 for true, 0 for false).

    Returns:
        dict: Dictionary mapping attack identifiers to lists of detected true positive nodes (as tuples).
    """
    # For large datasets, skip this function entirely to avoid OOM from loading node_to_path
    if len(nodes) > 100_000:
        print(f"Skipping get_detected_tps_node_level for large dataset ({len(nodes):,} nodes)")
        return {}

    # Ensure inputs are numpy arrays for easier manipulation
    scores = np.array(scores)
    y_truth = np.array(y_truth)

    # Create an array of indices to sort by scores
    indices = np.argsort(scores)[::-1]  # Sort in descending order

    # Sort scores, edges, and labels based on descending scores
    sorted_scores = scores[indices]
    sorted_nodes = [nodes[i] for i in indices]
    sorted_labels = y_truth[indices]

    # Find the cutoff k where no false positives (label 0) are included
    k = 0
    for i, label in enumerate(sorted_labels):
        if label == 0:  # Stop at the first false positive
            k = i
            break
        k = i + 1  # Include all true positives up to this point

    # Select the top k nodes (true positives)
    true_positive_nodes = sorted_nodes[:k]
    true_positive_indices = indices[:k]

    # For large datasets, skip loading full node_to_path to avoid OOM
    # Only load if we have a small number of true positives to display
    if len(true_positive_nodes) > 10000:
        # For large datasets, just use node IDs
        node_to_path = {}
    else:
        node_to_path = get_node_to_path_and_type(cfg)

    get_label = (
        lambda data: f"{data.get('path')}-{data.get('cmd')}".replace("None-", "")
        if data["type"] == "subject"
        else data.get("path")
    )

    # Map true positive nodes to their attacks
    attack_to_detected_nodes = defaultdict(list)
    for node in true_positive_nodes:
        attack = list(node2attack.get(node))[0]
        if attack is not None:  # Only include nodes with a valid attack mapping
            if int(node) in node_to_path:
                node_msg = get_label(node_to_path[int(node)]) + str(node)
            else:
                node_msg = f"node_{node}"
            attack_to_detected_nodes[attack].append(node_msg)

    return dict(attack_to_detected_nodes)


def get_ground_truth_nids(cfg):
    # ground_truth_nids, ground_truth_paths = [], {}
    # for file in cfg.dataset.ground_truth_relative_path:
    #     with open(os.path.join(cfg._ground_truth_dir, file), 'r') as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             node_uuid, node_labels, node_id = row[0], row[1], row[2]
    #             ground_truth_nids.append(int(node_id))
    #             ground_truth_paths[int(node_id)] = node_labels
    ground_truth_nids, ground_truth_paths, uuid_to_node_id = labelling.get_ground_truth(cfg)
    return set(ground_truth_nids), ground_truth_paths


def get_ground_truth_with_labels(cfg):
    """
    Get ground truth with attack/contaminated label distinction.

    Returns:
        attack_nids: set of node IDs labeled as 'attack' (strict malicious - attack chain)
        contaminated_nids: set of node IDs labeled as 'contaminated' (downstream effects)
        all_positive_nids: set of all positive node IDs (attack + contaminated)
        ground_truth_paths: dict mapping node_id -> path/label string
        excluded_nids: set of node IDs from excluded attack chains (to be masked in eval)
    """
    (
        attack_nids,
        contaminated_nids,
        all_positive_nids,
        ground_truth_paths,
        uuid_to_node_id,
        excluded_nids,
    ) = labelling.get_ground_truth_with_labels(cfg)
    return attack_nids, contaminated_nids, all_positive_nids, ground_truth_paths, excluded_nids


def get_ground_truth_uuid_to_node_id(cfg):
    # uuid_to_node_id = {}
    # for file in cfg.dataset.ground_truth_relative_path:
    #     with open(os.path.join(cfg._ground_truth_dir, file), 'r') as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             node_uuid, node_labels, node_id = row[0], row[1], row[2]
    #             uuid_to_node_id[node_uuid] = node_id
    ground_truth_nids, ground_truth_paths, uuid_to_node_id = labelling.get_ground_truth(cfg)
    return uuid_to_node_id


def compute_tw_labels(cfg):
    """
    Gets the malcious node IDs present in each time window.
    """
    out_path = cfg.preprocessing.build_graphs._tw_labels
    out_file = os.path.join(out_path, "tw_to_malicious_nodes.pkl")
    uuid_to_node_id = get_ground_truth_uuid_to_node_id(cfg)

    log("Computing time-window labels...")
    os.makedirs(out_path, exist_ok=True)

    t_to_node = labelling.get_t2malicious_node(cfg)
    # test_data = load_data_set(cfg, path=cfg.featurization.feat_inference._edge_embeds_dir, split="test")

    graph_dir = cfg.preprocessing.transformation._graphs_dir
    test_graphs = get_all_files_from_folders(graph_dir, cfg.dataset.test_files)

    num_found_event_labels = 0
    tw_to_malicious_nodes = defaultdict(list)
    for i, tw in enumerate(test_graphs):
        date = tw.split("/")[-1]
        start, end = (
            datetime_to_ns_time_US_handle_nano(date.split("~")[0]),
            datetime_to_ns_time_US_handle_nano(date.split("~")[1]),
        )

        for t, node_ids in t_to_node.items():
            if start < t < end:
                for node_id in node_ids:  # src, dst, or [src, dst] malicious nodes
                    tw_to_malicious_nodes[i].append(node_id)
                num_found_event_labels += 1

    log(f"Found {num_found_event_labels}/{len(t_to_node)} edge labels.")
    torch.save(tw_to_malicious_nodes, out_file)

    # Used to retrieve node ID from node raw UUID
    # node_labels_path = os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_events_relative_path)

    # uuid_to_node_id = get_ground_truth_uuid_to_node_id(cfg)

    # Create a mapping TW number => malicious node IDs
    for tw, nodes in tw_to_malicious_nodes.items():
        unique_nodes, counts = np.unique(nodes, return_counts=True)
        node_to_count = {node: count for node, count in zip(unique_nodes, counts)}
        log(f"TW {tw} -> {len(unique_nodes)} malicious nodes + {len(nodes)} malicious edges")

        node_to_count = {
            uuid_to_node_id[node_id]: count for node_id, count in node_to_count.items()
        }
        # pprint(node_to_count, width=1)
        tw_to_malicious_nodes[tw] = node_to_count

    return tw_to_malicious_nodes


def datetime_to_ns_time_US_handle_nano(nano_date_str):
    date = nano_date_str.split(".")[0]
    nanos = nano_date_str.split(".")[1]

    tz = pytz.timezone("US/Eastern")
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = str(timestamp).split(".")[0] + nanos
    return int(timeStamp)


def viz_graph(
    edge_index,
    edge_scores,
    node_scores,
    node_to_correct_pred,
    malicious_nodes,
    node_to_path_and_type,
    anomaly_threshold,
    out_dir,
    tw,
    cfg,
    n_hop,
    fuse_nodes,
):
    # On OpTC, the degree is too high so we remove 90% of non-malicious nodes
    # for visualization.
    # if dataset == "OPTC":
    #     OPTC_NODE_TO_VISUALIZE = 201  # Set the node viz manually
    #     idx = edge_index[0, :] == OPTC_NODE_TO_VISUALIZE

    #     if 1 not in y[idx]:
    #         return

    #     edge_index = edge_index[:, idx]
    #     edge_scores = edge_scores[idx]
    #     y = y[idx]

    #     indices_0 = np.where(y == 0)[0]
    #     indices_1 = np.where(y == 1)[0]

    #     selected_indices_0 = np.random.choice(
    #         indices_0, size=int(len(indices_0) * 0.1), replace=False
    #     )
    #     final_indices = np.concatenate((selected_indices_0, indices_1))
    #     np.random.shuffle(final_indices)

    #     edge_index = edge_index[:, final_indices]
    #     edge_scores = edge_scores[final_indices]
    #     y = y[final_indices]

    if edge_index.shape[0] != 2:
        edge_index = np.array([edge_index[:, 0], edge_index[:, 1]])

    if fuse_nodes:
        idx = 0
        merged_nodes = defaultdict(lambda: defaultdict(int))
        merged_edges = defaultdict(lambda: defaultdict(list))
        old_node_to_merged_node = defaultdict(list)
        for i, (src, dst, score) in enumerate(zip(edge_index[0], edge_index[1], edge_scores)):
            edge_tuple = []
            for node in [src, dst]:
                path = node_to_path_and_type[node]["path"]
                typ = node_to_path_and_type[node]["type"]

                if (path, typ) not in merged_nodes:
                    merged_nodes[(path, typ)] = {"idx": idx, "label": 0, "predicted": 0}
                    idx += 1
                edge_tuple.append(merged_nodes[(path, typ)]["idx"])
                old_node_to_merged_node[node] = merged_nodes[(path, typ)]["idx"]

                # If only one malicious node is present in the merged node, it is malicious
                merged_nodes[(path, typ)]["label"] = max(
                    merged_nodes[(path, typ)]["label"], int(node in malicious_nodes)
                )
                # I fonly one good prediction of the merged nodes is correct, we set predicted=1. If node not predicted, we set to -1
                merged_nodes[(path, typ)]["predicted"] = max(
                    merged_nodes[(path, typ)]["predicted"], int(node_to_correct_pred.get(node, -1))
                )

            merged_edges[tuple(edge_tuple)]["t"].append(i)
            merged_edges[tuple(edge_tuple)]["score"].append(score)

        new_edge_index = np.array(list(merged_edges.keys())).T
        merged_edge_scores = [np.max(d["score"]) for _, d in merged_edges.items()]
        edge_t = [f"{np.min(d['t'])}-{np.max(d['t'])}" for _, d in merged_edges.items()]

        # sorted_merged_nodes = dict(sorted(merged_nodes.items(), key=lambda item: item[1]))
        unique_nodes, unique_labels, unique_predicted, unique_paths, unique_types = (
            [],
            [],
            [],
            [],
            [],
        )
        for (path, typ), d in merged_nodes.items():
            unique_nodes.append(d["idx"])
            unique_labels.append(d["label"])
            unique_predicted.append(d["predicted"])
            unique_paths.append(path)
            unique_types.append(typ)

        source_nodes = malicious_nodes
        new_source_nodes = {old_node_to_merged_node[n] for n in source_nodes}

    else:
        # Flatten edge_index and map node IDs to a contiguous range starting from 0
        unique_nodes, new_edge_index = np.unique(edge_index.flatten(), return_inverse=True)
        new_edge_index = new_edge_index.reshape(edge_index.shape)
        unique_paths = [node_to_path_and_type[n]["path"] for n in unique_nodes]
        unique_types = [node_to_path_and_type[n]["type"] for n in unique_nodes]
        unique_labels = [n in malicious_nodes for n in unique_nodes]
        unique_predicted = [node_to_correct_pred.get(n, -1) for n in unique_nodes]
        edge_t = list(range(len(edge_index[0])))

        source_nodes = malicious_nodes
        source_node_map = {old: new for new, old in enumerate(unique_nodes)}
        new_source_nodes = [source_node_map.get(node, -1) for node in source_nodes]

    G = ig.Graph(edges=[tuple(e) for e in new_edge_index.T], directed=True)

    # Node attributes
    G.vs["original_id"] = unique_nodes
    G.vs["path"] = unique_paths
    G.vs["type"] = unique_types
    G.vs["shape"] = [
        "rectangle" if typ == "file" else "circle" if typ == "subject" else "triangle"
        for typ in unique_types
    ]

    G.vs["label"] = unique_labels
    G.vs["predicted"] = unique_predicted
    G.es["t"] = edge_t

    # Edge attributes
    G.es["anomaly_score"] = edge_scores

    # Find N-hop neighborhoods for the source nodes
    neighborhoods = set()
    for node in new_source_nodes:
        if node == -1:
            # Warning: one malicious node in {source_nodes} was not seen in the dataset ({new_source_nodes}).
            continue
        neighborhood = G.neighborhood(node, order=n_hop)
        neighborhoods.update(neighborhood)

    # Create a subgraph with only the n-hop neighborhoods
    subgraph = G.subgraph(neighborhoods)

    BENIGN = "#44BC"
    ATTACK = "#FF7E79"
    FAILURE = "red"
    SUCCESS = "green"

    visual_style = {}
    visual_style["bbox"] = (700, 700)
    visual_style["margin"] = 40
    visual_style["layout"] = subgraph.layout("kk")

    visual_style["vertex_size"] = 13
    visual_style["vertex_width"] = 13
    visual_style["vertex_label_dist"] = 1.3
    visual_style["vertex_label_size"] = 6
    visual_style["vertex_label_font"] = 1
    visual_style["vertex_color"] = [ATTACK if label else BENIGN for label in subgraph.vs["label"]]
    visual_style["vertex_label"] = subgraph.vs["path"]
    visual_style["vertex_frame_width"] = 2
    visual_style["vertex_frame_color"] = [
        "black" if predicted == -1 else SUCCESS if predicted else FAILURE
        for predicted in subgraph.vs["predicted"]
    ]

    visual_style["edge_curved"] = 0.1
    visual_style["edge_width"] = 1  # [3 if label else 1 for label in y_hat]
    visual_style["edge_color"] = (
        "gray"  # ["red" if label else "gray" for label in subgraph.es["y"]]
    )
    visual_style["edge_label"] = [
        f"s:{x:.2f}\nt:{t}" for x, t in zip(subgraph.es["anomaly_score"], subgraph.es["t"])
    ]
    visual_style["edge_label_size"] = 6
    visual_style["edge_label_color"] = "#888888"
    visual_style["edge_arrow_size"] = 8
    visual_style["edge_arrow_width"] = 8

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the graph using igraph
    plot = ig.plot(subgraph, target=ax, **visual_style)

    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=BENIGN, label="Benign"),
        mpatches.Patch(color=ATTACK, label="Attack"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="k",
            markersize=10,
            label="Subject",
            markeredgewidth=1,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="k",
            markersize=10,
            label="File",
            markeredgewidth=1,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="k",
            markersize=10,
            label="IP",
            markeredgewidth=1,
        ),
        mpatches.Patch(edgecolor=FAILURE, label="False Pos/Neg", facecolor="none"),
        mpatches.Patch(edgecolor=SUCCESS, label="True Pos/Neg", facecolor="none"),
    ]

    # Add legend to the plot
    ax.legend(handles=legend_handles, loc="upper right", fontsize="medium")

    # Save the plot with legend
    out_file = f"{n_hop}-hop_attack_graph_tw_{tw}"
    svg = os.path.join(out_dir, f"{out_file}.png")
    plt.savefig(svg)
    plt.close(fig)

    print(f"Graph {svg} saved, with attack nodes:\t {','.join([str(n) for n in source_nodes])}.")
    return {out_file: wandb.Image(svg)}


def compute_kmeans_labels(results, topk_K):
    nodes_to_score = sorted(
        [(node_id, d["score"]) for node_id, d in results.items()], key=lambda x: x[1]
    )
    nodes_to_score = np.array(nodes_to_score, dtype=object)
    score_values = nodes_to_score[:, 1].astype(float)

    last_N_scores = score_values[-topk_K:]
    last_N_nodes = nodes_to_score[-topk_K:]

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(last_N_scores.reshape(-1, 1))

    centroids = kmeans.cluster_centers_.flatten()
    highest_cluster_index = np.argmax(centroids)

    # Extract scores from the highest value cluster
    highest_value_cluster_indices = np.where(kmeans.labels_ == highest_cluster_index)[0]
    highest_value_cluster = last_N_nodes[highest_value_cluster_indices]

    # Extract scores and nodes from the highest cluster
    cluster_scores = highest_value_cluster[:, 1].astype(float)
    anomaly_nodes = highest_value_cluster[:, 0]

    for idx in highest_value_cluster_indices:
        global_idx = len(score_values) - topk_K + idx
        node_id = nodes_to_score[global_idx, 0]
        results[node_id]["y_hat"] = 1

    return results


def transform_attack2nodes_to_node2attacks(attack2nodes):
    node2attacks = defaultdict(set)
    for attack, nodes in attack2nodes.items():
        for node in nodes:
            node2attacks[node].add(attack)
    return dict(node2attacks)


def get_metrics_if_all_attacks_detected(pred_scores, nodes, attack_to_GPs):
    # Handle edge case: no nodes or no attacks
    if not pred_scores or not nodes or not attack_to_GPs:
        return 0, 0, 0.0, 0.0

    nodes_per_attack = [v["nids"] if isinstance(v, dict) else v for k, v in attack_to_GPs.items()]
    reverse_scores, reverse_nodes = zip(*sorted(zip(pred_scores, nodes), reverse=True))
    fps = 0
    detected_attacks = {}
    tps, fps = 0, 0
    total_attack_nodes = sum(len(nodes_set) for nodes_set in nodes_per_attack)

    for score, node in zip(reverse_scores, reverse_nodes):
        detected = False
        for i, nodes_set in enumerate(nodes_per_attack):
            if node in nodes_set:
                detected_attacks[i] = 1
                detected = True
        if len(detected_attacks) == len(nodes_per_attack):
            break
        if detected:
            tps += 1
        else:
            fps += 1

    precision = tps / (tps + fps + 1e-12)
    recall = tps / (total_attack_nodes + 1e-12)

    return fps, tps, precision, recall
