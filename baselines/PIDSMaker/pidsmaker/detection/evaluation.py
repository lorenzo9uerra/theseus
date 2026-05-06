import os

import wandb

from pidsmaker.detection.evaluation_methods import (
    edge_evaluation,
    node_evaluation,
    node_tw_evaluation,
    queue_evaluation,
    tw_evaluation,
)
from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    compute_tw_labels,
    listdir_sorted,
)
from pidsmaker.utils.utils import log


def standard_evaluation(cfg, evaluation_fn):
    test_losses_dir = os.path.join(cfg.detection.gnn_training._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.detection.gnn_training._edge_losses_dir, "val")

    tw_to_malicious_nodes = compute_tw_labels(cfg)
    has_validation_split = bool(cfg.dataset.val_files)

    # Find the best epoch based on VALIDATION performance
    best_epoch_info = find_best_epoch_from_validation(
        val_losses_dir, cfg, evaluation_fn, tw_to_malicious_nodes
    )
    best_model_epoch_dir = best_epoch_info["best_epoch_dir"]

    if has_validation_split:
        log(f"Best epoch selected from validation: {best_model_epoch_dir}", pre_return_line=True)
        log(
            f"Validation metrics - ADP: {best_epoch_info['adp_score']:.3f}, Discrimination: {best_epoch_info['discrimination']:.3f}, AP: {best_epoch_info['ap']:.3f}, MCC: {best_epoch_info['mcc']:.3f}"
        )
    else:
        log(
            f"No validation split configured; using the last epoch ({best_model_epoch_dir}) and train-calibration losses.",
            pre_return_line=True,
        )

    # Now evaluate ONLY the best epoch on the test set
    log(
        f"[@{best_model_epoch_dir}] - Test Evaluation (best epoch from validation)",
        pre_return_line=True,
    )

    test_tw_path = os.path.join(test_losses_dir, best_model_epoch_dir)
    val_tw_path = os.path.join(val_losses_dir, best_model_epoch_dir)

    stats = evaluation_fn(
        val_tw_path,
        test_tw_path,
        best_model_epoch_dir,
        cfg,
        tw_to_malicious_nodes=tw_to_malicious_nodes,
    )
    log(f"[@{best_model_epoch_dir}] - Test Stats")
    for k, v in stats.items():
        log(f"{k}: {v}")

    stats["epoch"] = int(best_model_epoch_dir.split("_")[-1])

    out_dir = cfg.detection.evaluation._precision_recall_dir
    save_files_to_wandb = cfg._experiment != "uncertainty"

    if save_files_to_wandb:
        scores = os.path.join(out_dir, f"scores_{best_model_epoch_dir}.png")
        if os.path.exists(scores):
            stats["scores_img"] = wandb.Image(scores)

        adp = os.path.join(out_dir, f"adp_curve_{best_model_epoch_dir}.png")
        if os.path.exists(adp):
            stats["adp_img"] = wandb.Image(adp)

        seen_scores = os.path.join(out_dir, f"seen_score_{best_model_epoch_dir}.png")
        if os.path.exists(seen_scores):
            stats["seen_scores_img"] = wandb.Image(seen_scores)

        discrim = os.path.join(out_dir, f"discrim_curve_{best_model_epoch_dir}.png")
        if os.path.exists(discrim):
            stats["discrim_img"] = wandb.Image(discrim)

        # Anomaly score distribution plot
        anomaly_dist = os.path.join(out_dir, f"score_distribution_{best_model_epoch_dir}.png")
        if os.path.exists(anomaly_dist):
            stats["anomaly_distribution_img"] = wandb.Image(anomaly_dist)

        # Save the best run artifacts (only if heavy plots were generated)
        if "neat_scores_img_file" in stats:
            wandb.save(stats["neat_scores_img_file"], out_dir)

    wandb.log(stats)

    return stats


def find_best_epoch_from_validation(val_losses_dir, cfg, evaluation_fn, tw_to_malicious_nodes):
    """
    Evaluate all epochs on the validation set and return the best epoch info.
    This prevents data leakage by using only validation performance for model selection.
    """
    if not os.path.exists(val_losses_dir):
        # Fallback to epoch 0 if no validation directory exists
        return {
            "best_epoch_dir": "epoch_0",
            "adp_score": float("-inf"),
            "discrimination": float("-inf"),
            "ap": float("-inf"),
            "mcc": float("-inf"),
        }

    sorted_files = listdir_sorted(val_losses_dir)
    if not cfg.dataset.val_files:
        return {
            "best_epoch_dir": sorted_files[-1] if sorted_files else "epoch_0",
            "adp_score": float("-inf"),
            "discrimination": float("-inf"),
            "ap": float("-inf"),
            "mcc": float("-inf"),
        }

    best_metrics = {
        "adp_score": float("-inf"),
        "discrimination": float("-inf"),
        "ap": float("-inf"),
        "mcc": float("-inf"),
        "best_epoch_dir": None,
    }

    best_model_selection = cfg.detection.evaluation.best_model_selection
    if best_model_selection == "best_pr_auc":
        best_model_selection = "best_ap"

    log("Evaluating validation set to find best epoch...")
    for model_epoch_dir in sorted_files:
        log(f"[@{model_epoch_dir}] - Validation Evaluation")

        val_tw_path = os.path.join(val_losses_dir, model_epoch_dir)

        # For validation evaluation, we use the same val path for both arguments
        # since we're only computing metrics on validation data
        stats = evaluation_fn(
            val_tw_path,
            val_tw_path,  # Use val for both since we're evaluating on val
            model_epoch_dir,
            cfg,
            tw_to_malicious_nodes=tw_to_malicious_nodes,
        )

        # Select best epoch based on validation performance
        if best_model_selection == "best_adp":
            condition = (stats["adp_score"] > best_metrics["adp_score"]) or (
                stats["adp_score"] == best_metrics["adp_score"]
                and stats["discrimination"] > best_metrics["discrimination"]
            )
        elif best_model_selection == "best_discrimination":
            condition = stats["discrimination"] > best_metrics["discrimination"]
        elif best_model_selection == "best_ap":
            condition = stats["ap"] > best_metrics["ap"]
        else:
            raise ValueError(f"Invalid best model selection {best_model_selection}")

        if condition:
            best_metrics["adp_score"] = stats["adp_score"]
            best_metrics["discrimination"] = stats["discrimination"]
            best_metrics["ap"] = stats["ap"]
            best_metrics["mcc"] = stats.get("mcc", 0.0)
            best_metrics["best_epoch_dir"] = model_epoch_dir
            log(
                f"  -> New best: ADP={stats['adp_score']:.3f}, Discrimination={stats['discrimination']:.3f}, AP={stats['ap']:.3f}, MCC={stats.get('mcc', 0.0):.3f}"
            )

    if best_metrics["best_epoch_dir"] is None:
        best_metrics["best_epoch_dir"] = sorted_files[0] if sorted_files else "epoch_0"

    return best_metrics


def main(cfg):
    method = cfg.detection.evaluation.used_method.strip()
    if method == "node_evaluation":
        return standard_evaluation(cfg, evaluation_fn=node_evaluation.main)
    elif method == "tw_evaluation":
        return standard_evaluation(cfg, evaluation_fn=tw_evaluation.main)
    elif method == "node_tw_evaluation":
        return standard_evaluation(cfg, evaluation_fn=node_tw_evaluation.main)
    elif method == "queue_evaluation":
        return queue_evaluation.main(cfg)
    elif method == "edge_evaluation":
        return standard_evaluation(cfg, evaluation_fn=edge_evaluation.main)
    else:
        raise ValueError(f"Invalid evaluation method {cfg.detection.evaluation.used_method}")
