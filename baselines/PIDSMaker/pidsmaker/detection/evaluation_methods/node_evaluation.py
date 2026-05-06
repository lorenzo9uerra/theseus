import gc
import os
from collections import defaultdict

import pandas as pd
import torch

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    classifier_evaluation,
    compute_discrimination_score,
    compute_discrimination_tp,
    compute_kmeans_labels,
    datetime_to_ns_time_US_handle_nano,
    get_best_mcc_threshold,
    get_detected_tps_node_level,
    get_ground_truth_nids,
    get_ground_truth_with_labels,
    get_max_benign_threshold,
    get_metrics_if_all_attacks_detected,
    get_threshold,
    plot_anomaly_score_distribution,
    plot_detected_attacks_vs_precision,
    plot_discrimination_metric,
    plot_scores_neat,
    plot_scores_with_paths_node_level,
    reduce_losses_to_score,
    transform_attack2nodes_to_node2attacks,
)
from pidsmaker.utils.labelling import get_attack_chains, get_GP_of_each_attack
from pidsmaker.utils.utils import (
    get_all_files_from_folders,
    get_node_to_path_and_type,
    listdir_sorted,
    log,
    log_tqdm,
)


def get_node_predictions(val_tw_path, test_tw_path, cfg, **kwargs):
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Loading data from {test_tw_path}...")

    threshold_method = cfg.detection.evaluation.node_evaluation.threshold_method

    # Handle best_mcc threshold method
    if threshold_method.strip() == "best_mcc":
        use_dst = cfg.detection.evaluation.node_evaluation.use_dst_node_loss
        thr, val_mcc = get_best_mcc_threshold(
            val_tw_path, ground_truth_nids, cfg, use_dst_node_loss=use_dst
        )
        log(f"Using best_mcc threshold: {thr:.5f} (validation MCC: {val_mcc:.4f})")
    elif threshold_method.strip() == "max_benign_val_score":
        use_dst = cfg.detection.evaluation.node_evaluation.use_dst_node_loss
        thr = get_max_benign_threshold(
            val_tw_path, cfg, use_dst_node_loss=use_dst
        )
        log(f"Using max_benign_val_score threshold: {thr:.5f}")
    else:
        # Always use validation set to compute threshold (no data leakage)
        thr = get_threshold(val_tw_path, threshold_method)
        log(f"Threshold: {thr:.3f}")

    node_to_losses = defaultdict(list)
    node_to_max_loss_tw = {}
    node_to_max_loss = defaultdict(int)

    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(log_tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient="records")
        for line in df:
            srcnode = line["srcnode"]
            dstnode = line["dstnode"]
            loss = line["loss"]

            # Scores
            node_to_losses[srcnode].append(loss)
            if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                node_to_losses[dstnode].append(loss)

            # If max-val thr is used, we want to keep track when the node with max loss happens
            if loss > node_to_max_loss[srcnode]:
                node_to_max_loss[srcnode] = loss
                node_to_max_loss_tw[srcnode] = tw
            if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                if loss > node_to_max_loss[dstnode]:
                    node_to_max_loss[dstnode] = loss
                    node_to_max_loss_tw[dstnode] = tw

    # For plotting the scores of seen and unseen nodes
    graph_dir = cfg.preprocessing.transformation._graphs_dir
    train_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.train_files)

    train_node_set = set()
    for train_path in train_set_paths:
        train_graph = torch.load(train_path)
        train_node_set |= set(train_graph.nodes())

    use_kmeans = cfg.detection.evaluation.node_evaluation.use_kmeans
    results = defaultdict(dict)
    for node_id, losses in node_to_losses.items():
        pred_score = reduce_losses_to_score(
            losses, cfg.detection.evaluation.node_evaluation.threshold_method
        )

        results[node_id]["score"] = pred_score
        results[node_id]["tw_with_max_loss"] = node_to_max_loss_tw.get(node_id, -1)
        results[node_id]["y_true"] = int(node_id in ground_truth_nids)
        results[node_id]["is_seen"] = int(str(node_id) in train_node_set)

        if use_kmeans:  # in this mode, we add the label after
            results[node_id]["y_hat"] = 0
        else:
            results[node_id]["y_hat"] = int(pred_score > thr)

    if use_kmeans:
        results = compute_kmeans_labels(
            results,
            topk_K=cfg.detection.evaluation.node_evaluation.kmeans_top_K,
        )
    return results, thr


def get_node_predictions_node_level(val_tw_path, test_tw_path, cfg, **kwargs):
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Loading data from {test_tw_path}...")

    threshold_method = cfg.detection.evaluation.node_evaluation.threshold_method

    # Handle best_mcc threshold method
    if threshold_method.strip() == "best_mcc":
        thr, val_f1 = get_best_mcc_threshold(
            val_tw_path, ground_truth_nids, cfg, use_dst_node_loss=False
        )
        log(f"Using best_mcc threshold: {thr:.5f} (validation MCC: {val_f1:.4f})")
    elif threshold_method.strip() == "max_benign_val_score":
        thr = get_max_benign_threshold(
            val_tw_path, cfg, use_dst_node_loss=False
        )
        log(f"Using max_benign_val_score threshold: {thr:.5f}")
    else:
        # Always use validation set to compute threshold (no data leakage)
        thr = get_threshold(val_tw_path, threshold_method)
        log(f"Threshold: {thr:.3f}")

    node_to_values = defaultdict(lambda: defaultdict(list))
    node_to_max_loss_tw = {}
    node_to_max_loss = defaultdict(int)

    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(log_tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient="records")
        for line in df:
            node = line["node"]
            loss = line["loss"]

            node_to_values[node]["loss"].append(loss)
            node_to_values[node]["tw"].append(tw)

            if "threatrace_score" in line:
                node_to_values[node]["threatrace_score"].append(line["threatrace_score"])
            if "correct_pred" in line:
                node_to_values[node]["correct_pred"].append(line["correct_pred"])
            if "flash_score" in line:
                node_to_values[node]["flash_score"].append(line["flash_score"])
            if "magic_score" in line:
                node_to_values[node]["magic_score"].append(line["magic_score"])

            if loss > node_to_max_loss[node]:
                node_to_max_loss[node] = loss
                node_to_max_loss_tw[node] = tw

    # For plotting the scores of seen and unseen nodes
    graph_dir = cfg.preprocessing.transformation._graphs_dir
    train_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.train_files)

    train_node_set = set()
    for train_path in train_set_paths:
        train_graph = torch.load(train_path)
        train_node_set |= set(train_graph.nodes())

    use_kmeans = cfg.detection.evaluation.node_evaluation.use_kmeans
    results = defaultdict(dict)
    for node_id, losses in node_to_values.items():
        threatrace_label = 0
        flash_label = 0
        detected_tw = None
        if cfg.detection.evaluation.node_evaluation.threshold_method == "threatrace":
            max_score = 0
            pred_score = max(losses["threatrace_score"])

            for score, node_type_pred, tw in zip(
                losses["threatrace_score"], losses["correct_pred"], losses["tw"]
            ):
                if score > thr and node_type_pred and score > max_score:
                    threatrace_label = 1
                    max_score = score
                    detected_tw = tw

        elif cfg.detection.evaluation.node_evaluation.threshold_method == "flash":
            max_score = 0
            pred_score = max(losses["flash_score"])

            for score, node_type_pred, tw in zip(
                losses["flash_score"], losses["correct_pred"], losses["tw"]
            ):
                if score > thr and node_type_pred and score > max_score:
                    flash_label = 1
                    max_score = score
                    detected_tw = tw

        elif cfg.detection.evaluation.node_evaluation.threshold_method == "magic":
            max_score = 0
            pred_score = max(losses["magic_score"])

            for score, tw in zip(losses["magic_score"], losses["tw"]):
                if score > thr and score > max_score:
                    flash_label = 1
                    max_score = score
                    detected_tw = tw

        else:
            pred_score = reduce_losses_to_score(
                losses["loss"],
                cfg.detection.evaluation.node_evaluation.threshold_method,
            )

        results[node_id]["score"] = pred_score
        results[node_id]["tw_with_max_loss"] = node_to_max_loss_tw.get(node_id, -1)
        results[node_id]["y_true"] = int(node_id in ground_truth_nids)
        results[node_id]["is_seen"] = int(str(node_id) in train_node_set)

        # We need the detected TW range to check if the detected node spans in an attack TW
        detected_tw = detected_tw or node_to_max_loss_tw.get(node_id, None)
        if detected_tw is not None:
            results[node_id]["time_range"] = [
                datetime_to_ns_time_US_handle_nano(tw) for tw in filelist[detected_tw].split("~")
            ]
        else:
            results[node_id]["time_range"] = None

        if use_kmeans:  # in this mode, we add the label after
            results[node_id]["y_hat"] = 0
        else:
            if cfg.detection.evaluation.node_evaluation.threshold_method == "threatrace":
                results[node_id]["y_hat"] = threatrace_label
            elif cfg.detection.evaluation.node_evaluation.threshold_method == "flash":
                results[node_id]["y_hat"] = flash_label
            else:
                results[node_id]["y_hat"] = int(pred_score > thr)

    if use_kmeans:
        results = compute_kmeans_labels(
            results,
            topk_K=cfg.detection.evaluation.node_evaluation.kmeans_top_K,
        )
    return results, thr


def analyze_false_positives(
    y_truth, y_preds, pred_scores, max_val_loss_tw, nodes, tw_to_malicious_nodes
):
    fp_indices = [i for i, (true, pred) in enumerate(zip(y_truth, y_preds)) if pred and not true]
    malicious_tws = set(tw_to_malicious_nodes.keys())
    num_fps_in_malicious_tw = 0

    for i in fp_indices:
        is_in_malicious_tw = max_val_loss_tw[i] in malicious_tws
        num_fps_in_malicious_tw += int(is_in_malicious_tw)

    fp_in_malicious_tw_ratio = (
        num_fps_in_malicious_tw / len(fp_indices) if len(fp_indices) > 0 else float("nan")
    )
    return fp_in_malicious_tw_ratio


def evaluate_single_level(
    nodes,
    y_truth,
    y_preds,
    pred_scores,
    max_val_loss_tw,
    node2attacks,
    attack_to_GPs,
    node_to_path,
    tw_to_malicious_nodes,
    out_dir,
    model_epoch_dir,
    cfg,
    thr,
):
    attack_to_TPs = defaultdict(int)

    level_name = "Strict (attack only)"
    log(f"\n{'=' * 60}")
    log(f"Evaluation Level: {level_name}")
    log(f"{'=' * 60}")

    log(f"Analysis of malicious nodes ({level_name}):")
    count = 0
    for i, nid in enumerate(nodes):
        y_true = y_truth[i]
        y_hat = y_preds[i]
        score = pred_scores[i]

        if y_true == 1 and count < 20:
            count += 1
            log(
                f"-> Malicious node {nid:<7}: loss={score:.3f} | is TP:"
                + (" ✅ " if y_true == y_hat else " ❌ ")
                + (node_to_path[nid]["path"] if nid in node_to_path else "unknown")
            )

            if y_hat:
                for att, d in attack_to_GPs.items():
                    if nid in d["nids"]:
                        attack_to_TPs[att] += 1

    stats = classifier_evaluation(y_truth, y_preds, pred_scores)

    num_nodes = len(nodes)
    skip_heavy_plots = num_nodes > 1_000_000

    if skip_heavy_plots:
        log(f"Skipping heavy plots for large dataset ({num_nodes:,} nodes)")

    adp_img_file = os.path.join(out_dir, f"adp_curve_{model_epoch_dir}.png")
    discrim_img_file = os.path.join(out_dir, f"discrim_curve_{model_epoch_dir}.png")
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    neat_scores_img_file = os.path.join(out_dir, f"neat_scores_{model_epoch_dir}.svg")

    log(f"Saving figures to {out_dir}...")
    adp_score = plot_detected_attacks_vs_precision(
        pred_scores, nodes, node2attacks, y_truth, adp_img_file
    )

    gc.collect()

    discrim_scores = compute_discrimination_score(pred_scores, nodes, node2attacks, y_truth)

    if not skip_heavy_plots:
        plot_discrimination_metric(pred_scores, y_truth, discrim_img_file)

    discrim_tp = compute_discrimination_tp(pred_scores, nodes, node2attacks, y_truth)

    if not skip_heavy_plots:
        plot_scores_with_paths_node_level(
            pred_scores,
            y_truth,
            nodes,
            max_val_loss_tw,
            tw_to_malicious_nodes,
            node2attacks,
            scores_img_file,
            cfg,
            thr,
        )
        plot_scores_neat(pred_scores, y_truth, nodes, node2attacks, neat_scores_img_file, thr)

        score_dist_img_file = os.path.join(out_dir, f"score_distribution_{model_epoch_dir}.png")
        plot_anomaly_score_distribution(
            pred_scores,
            y_truth,
            score_dist_img_file,
            threshold=thr,
            nodes=nodes,
            node_to_attack_ids=node2attacks,
        )

    stats["adp_score"] = round(adp_score, 3)
    for k, v in discrim_scores.items():
        stats[k] = round(v, 4)
    stats = {**stats, **discrim_tp}

    attack2tps = get_detected_tps_node_level(pred_scores, nodes, node2attacks, y_truth, cfg)
    for attack, detected_tps in attack2tps.items():
        stats[f"tps_{attack}"] = str(detected_tps)

    if not skip_heavy_plots:
        stats["neat_scores_img_file"] = neat_scores_img_file

    fp_in_malicious_tw_ratio = analyze_false_positives(
        y_truth,
        y_preds,
        pred_scores,
        max_val_loss_tw,
        nodes,
        tw_to_malicious_nodes,
    )
    stats["fp_in_malicious_tw_ratio"] = round(fp_in_malicious_tw_ratio, 3)

    log(f"TPs per attack ({level_name}):")
    for att, tps in attack_to_TPs.items():
        log(f"attack {att}: {tps}")

    stats["percent_detected_attacks"] = (
        round(len(attack_to_TPs) / len(attack_to_GPs), 2) if len(attack_to_GPs) > 0 else 0
    )

    fps, tps, precision, recall = get_metrics_if_all_attacks_detected(
        pred_scores, nodes, attack_to_GPs
    )
    stats["fps_if_all_attacks_detected"] = fps
    stats["tps_if_all_attacks_detected"] = tps
    stats["precision_if_all_attacks_detected"] = precision
    stats["recall_if_all_attacks_detected"] = recall

    gc.collect()

    return stats


def main(
    val_tw_path,
    test_tw_path,
    model_epoch_dir,
    cfg,
    tw_to_malicious_nodes,
    **kwargs,
):
    if cfg._is_node_level:
        get_preds_fn = get_node_predictions_node_level
    else:
        get_preds_fn = get_node_predictions

    results, thr = get_preds_fn(cfg=cfg, val_tw_path=val_tw_path, test_tw_path=test_tw_path)

    # save results for future checking
    os.makedirs(cfg.detection.evaluation._results_dir, exist_ok=True)
    results_save_dir = os.path.join(cfg.detection.evaluation._results_dir, "results.pth")
    torch.save(results, results_save_dir)
    log(f"Results saved to {results_save_dir}")

    node_to_path = {}

    out_dir = cfg.detection.evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)

    attack_to_GPs = get_GP_of_each_attack(cfg)

    attack_nids, contaminated_nids, all_positive_nids, _, excluded_nids = (
        get_ground_truth_with_labels(cfg)
    )

    log(f"\nGround Truth Summary:")
    log(f"  Attack nodes (attack chain): {len(attack_nids)}")
    log(f"  Contaminated nodes (downstream): {len(contaminated_nids)}")
    log(f"  Total positive nodes: {len(all_positive_nids)}")
    if excluded_nids:
        log(f"  Excluded nodes (bad neighborhood): {len(excluded_nids)}")

    log("Loading node types for process-only evaluation...")
    node_to_info = get_node_to_path_and_type(cfg)

    nodes, y_preds, pred_scores, max_val_loss_tw, is_seen = [], [], [], [], []
    num_excluded = 0
    num_non_process = 0

    for nid, result in results.items():
        # Skip nodes from excluded attack chains (e.g., wwtawwtal_bad_neighborhood)
        # These are completely removed from evaluation
        if nid in excluded_nids:
            num_excluded += 1
            continue

        # Only evaluate process nodes (ground truth only labels processes)
        if nid not in node_to_info or node_to_info[nid]["type"] != "subject":
            num_non_process += 1
            continue

        nodes.append(nid)
        y_preds.append(result["y_hat"])
        pred_scores.append(result["score"])
        max_val_loss_tw.append(result["tw_with_max_loss"])
        is_seen.append(result["is_seen"])

    if num_excluded > 0:
        log(f"  Excluded {num_excluded} nodes from evaluation (from excluded attack chains)")

    log(f"\n{'=' * 60}")
    log("PROCESS-ONLY EVALUATION MODE")
    log(f"{'=' * 60}")
    log(f"  Total nodes in results: {len(results)}")
    log(f"  Excluded (attack chains): {num_excluded}")
    log(f"  Filtered out (files/sockets): {num_non_process}")
    log(f"  Evaluating (processes only): {len(nodes)}")
    log(f"{'=' * 60}")

    nodes_set = set(nodes)
    attack_nodes_in_eval = attack_nids & nodes_set
    contaminated_nodes_in_eval = contaminated_nids & nodes_set
    all_positive_in_eval = all_positive_nids & nodes_set

    log(f"\nGround Truth Overlap with Evaluated Nodes:")
    log(f"  Attack nodes in evaluation: {len(attack_nodes_in_eval)} / {len(attack_nids)}")
    log(
        f"  Contaminated nodes in evaluation: {len(contaminated_nodes_in_eval)} / {len(contaminated_nids)}"
    )
    log(
        f"  Total positive nodes in evaluation: {len(all_positive_in_eval)} / {len(all_positive_nids)}"
    )

    if len(all_positive_in_eval) == 0:
        log("\nWARNING: No ground truth nodes found in evaluation set")

        # Check a sample of ground truth nodes
        sample_gt_nodes = list(attack_nids)[:5]
        for gt_node in sample_gt_nodes:
            in_results = gt_node in results
            in_node_info = gt_node in node_to_info
            node_type = (
                node_to_info.get(gt_node, {}).get("type", "unknown")
                if in_node_info
                else "not found"
            )
            log(
                f"    Sample GT node {gt_node}: in_results={in_results}, in_node_info={in_node_info}, type={node_type}"
            )

    y_truth_full = [int(nid in attack_nids) for nid in nodes]
    mask_strict = [nid not in contaminated_nids for nid in nodes]
    node_to_path = node_to_info

    del results
    gc.collect()

    attack_chains = get_attack_chains(cfg)
    attack2nodes = {}
    if attack_chains:
        for chain_name, chain_data in attack_chains.items():
            if chain_data["attack_nids"]:
                attack2nodes[chain_name] = chain_data["attack_nids"]
        log(
            f"Built node2attacks from {len(attack_chains)} attack chains: {list(attack_chains.keys())}"
        )
    else:
        log("WARNING: No attack chains found, falling back to attack_to_GPs for node2attacks")
        for attack_id, metadata in attack_to_GPs.items():
            attack_label = metadata.get("attack_chain", attack_id)
            attack2nodes[attack_label] = metadata["nids"]

    node2attacks = transform_attack2nodes_to_node2attacks(attack2nodes)

    nodes_in_node2attacks = sum(1 for n in nodes if n in node2attacks)
    log(f"Nodes in node2attacks: {nodes_in_node2attacks} / {len(nodes)} evaluated nodes")

    nodes_strict = [nid for nid, keep in zip(nodes, mask_strict) if keep]
    y_truth_strict = [int(nid in attack_nids) for nid in nodes_strict]
    y_preds_strict = [pred for pred, keep in zip(y_preds, mask_strict) if keep]
    pred_scores_strict = [score for score, keep in zip(pred_scores, mask_strict) if keep]
    max_val_loss_tw_strict = [tw for tw, keep in zip(max_val_loss_tw, mask_strict) if keep]

    attack_to_GPs_strict = {
        att: data for att, data in attack_to_GPs.items() if data.get("nids")
    }

    log(f"\nStrict Evaluation Set:")
    log(f"  Nodes after masking contaminated: {len(nodes_strict)} (from {len(nodes)})")
    log(f"  Contaminated nodes excluded: {len(contaminated_nodes_in_eval)}")
    log(f"  Attack chains in evaluation: {len(attack_to_GPs_strict)}")

    stats_strict = evaluate_single_level(
        nodes=nodes_strict,
        y_truth=y_truth_strict,
        y_preds=y_preds_strict,
        pred_scores=pred_scores_strict,
        max_val_loss_tw=max_val_loss_tw_strict,
        node2attacks=node2attacks,
        attack_to_GPs=attack_to_GPs_strict,
        node_to_path=node_to_path,
        tw_to_malicious_nodes=tw_to_malicious_nodes,
        out_dir=out_dir,
        model_epoch_dir=model_epoch_dir,
        cfg=cfg,
        thr=thr,
    )

    final_stats = {}
    for k, v in stats_strict.items():
        final_stats[k] = v
        final_stats[f"final_{k}"] = v

    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")
    scores_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.pkl")

    torch.save(final_stats, stats_file)

    torch.save(
        {
            "pred_scores": pred_scores_strict,
            "y_preds": y_preds_strict,
            "y_truth": y_truth_strict,
            "mask_evaluated": mask_strict,
            "nodes": nodes_strict,
            "node2attacks": node2attacks,
            "attack_nids": attack_nids,
            "contaminated_nids": contaminated_nids,
            "all_nodes_before_masking": nodes,
            "y_truth_before_masking": y_truth_full,
        },
        scores_file,
    )

    final_stats["scores_file"] = scores_file

    log(f"\n{'=' * 60}")
    log("STRICT EVALUATION SUMMARY")
    log(f"{'=' * 60}")
    log(f"  - Precision: {final_stats['final_precision']:.4f}")
    log(f"  - Recall: {final_stats['final_recall']:.4f}")
    log(f"  - F-Score: {final_stats['final_fscore']:.4f}")
    log(f"  - ADP Score: {final_stats['final_adp_score']:.4f}")
    log(f"  - Discrimination: {final_stats['final_discrimination']:.4f}")
    log(f"{'=' * 60}")

    return final_stats
