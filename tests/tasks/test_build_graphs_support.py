from types import SimpleNamespace

import torch

from tasks.build_graphs_support import (
    EXCLUDED_ATTACK_CLASSES,
    determine_split,
    filter_ground_truth,
    format_event_day_label,
    get_malicious_nodes,
    get_selected_days,
    relabel_graphs,
)


def make_config():
    return SimpleNamespace(
        dataset_info=SimpleNamespace(
            train_days=[3, 4],
            val_days=[5],
            test_days=[6, 7],
            year_month="2018-04",
        )
    )


def test_determine_split_uses_dataset_days():
    config = make_config()

    assert determine_split(3, config) == "train"
    assert determine_split(5, config) == "val"
    assert determine_split(7, config) == "test"
    assert determine_split(99, config) == "train"


def test_selected_days_and_day_label_are_stable():
    config = make_config()

    assert get_selected_days(config) == [3, 4, 5, 6, 7]
    assert format_event_day_label(config, 3) == "2018-04-03"


def test_ground_truth_helpers_filter_and_collect_malicious_nodes():
    excluded_attack = next(iter(EXCLUDED_ATTACK_CLASSES))
    ground_truth = {
        "kept": {"nids": [1, 2], "contaminated_nids": [3]},
        excluded_attack: {"nids": [10], "contaminated_nids": [11]},
    }

    filtered_ground_truth = filter_ground_truth(ground_truth)

    assert excluded_attack not in filtered_ground_truth
    assert get_malicious_nodes(filtered_ground_truth) == {1, 2, 3}


def test_relabel_graphs_updates_labels_from_original_node_ids():
    graph = SimpleNamespace(
        original_n_id=torch.tensor([10, 11, 12], dtype=torch.long),
        y=torch.zeros(3, dtype=torch.long),
    )
    graphs = {"train": [graph], "val": [], "test": []}

    relabel_graphs(graphs, {10, 12})

    assert graph.y.tolist() == [1, 0, 1]
