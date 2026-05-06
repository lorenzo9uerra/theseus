from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tasks.build_graphs import build_graphs, create_graph_window


def make_config():
    return SimpleNamespace(
        cache_dir="/tmp/cache",
        dataset="THEIA_E3",
        dataset_info=SimpleNamespace(train_days=[1], val_days=[2], test_days=[3]),
        use_node_degrees=True,
        bidirectional_edges=True,
        use_fused_edge_count=True,
    )


def test_create_graph_window_builds_expected_graph_features():
    config = make_config()
    edge_arrays = (
        np.array([1, 2], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
        np.array([100, 200], dtype=np.int64),
        ["READ", "WRITE"],
        np.array([2, 1], dtype=np.int32),
    )
    node_embeddings = {
        "1": torch.tensor([1.0, 0.0]),
        "2": torch.tensor([0.0, 1.0]),
        "3": torch.tensor([1.0, 1.0]),
    }
    edge_type_to_onehot = {"READ": object(), "WRITE": object()}

    graph, split = create_graph_window(
        edge_arrays=edge_arrays,
        day=2,
        config=config,
        node_embeddings=node_embeddings,
        edge_type_to_onehot=edge_type_to_onehot,
        malicious_nodes={3},
    )

    assert split == "val"
    assert graph.original_n_id.tolist() == [1, 2, 3]
    assert graph.y.tolist() == [0, 0, 1]
    assert torch.allclose(
        graph.x[:, :2],
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        graph.x[:, 2], torch.tensor(np.log1p([1, 2, 1]), dtype=torch.float32)
    )
    assert graph.edge_index.tolist() == [[0, 1, 1, 2], [1, 2, 0, 1]]
    assert torch.allclose(
        graph.edge_attr,
        torch.tensor(
            [
                [1.0, 0.0, np.log1p(2.0)],
                [0.0, 1.0, np.log1p(1.0)],
                [1.0, 0.0, np.log1p(2.0)],
                [0.0, 1.0, np.log1p(1.0)],
            ],
            dtype=torch.float32,
        ),
    )
    assert graph.t.tolist() == [100, 200, 100, 200]


def test_build_graphs_build_path_wires_dependencies(monkeypatch):
    config = make_config()
    expected_graphs = {"train": ["g1"], "val": [], "test": ["g2"]}
    original_ground_truth = {
        "kept": {"nids": [10], "contaminated_nids": [20]},
        "wwtawwtal_bad_neighborhood": {"nids": [99], "contaminated_nids": [98]},
    }
    call_log = []
    saved_payload = {}

    monkeypatch.setattr(
        "tasks.build_graphs.load_graph_cache",
        lambda cache_dir, dataset_name, cfg: (None, None),
    )

    def fake_get_ground_truth(cfg):
        call_log.append("get_ground_truth")
        return original_ground_truth

    def fake_fetch_node_metadata(cfg):
        call_log.append("fetch_node_metadata")
        return {"10": ["process", "proc"]}

    def fake_get_training_nodes_from_csv(cfg):
        call_log.append("get_training_nodes_from_csv")
        return {"10"}

    def fake_train_word2vec(node_metadata, train_nodes, cfg):
        call_log.append(("train_word2vec", node_metadata, train_nodes))
        return {"path": "model"}

    def fake_gen_edge_fused_tw(node_metadata, cfg, word2vec_models, malicious_nodes):
        call_log.append(("gen_edge_fused_tw", node_metadata, word2vec_models))
        assert malicious_nodes == {10, 20}
        return expected_graphs

    def fake_save_graph_cache(graphs, ground_truth, cache_dir, dataset_name, cfg):
        saved_payload["graphs"] = graphs
        saved_payload["ground_truth"] = ground_truth
        saved_payload["cache_dir"] = cache_dir
        saved_payload["dataset_name"] = dataset_name

    monkeypatch.setattr("tasks.build_graphs.get_ground_truth", fake_get_ground_truth)
    monkeypatch.setattr(
        "tasks.build_graphs.fetch_node_metadata", fake_fetch_node_metadata
    )
    monkeypatch.setattr(
        "tasks.build_graphs.get_training_nodes_from_csv",
        fake_get_training_nodes_from_csv,
    )
    monkeypatch.setattr("tasks.build_graphs.train_word2vec", fake_train_word2vec)
    monkeypatch.setattr("tasks.build_graphs.gen_edge_fused_tw", fake_gen_edge_fused_tw)
    monkeypatch.setattr("tasks.build_graphs.save_graph_cache", fake_save_graph_cache)

    graphs, filtered_ground_truth = build_graphs(config)

    assert graphs == expected_graphs
    assert filtered_ground_truth == {"kept": {"nids": [10], "contaminated_nids": [20]}}
    assert saved_payload == {
        "graphs": expected_graphs,
        "ground_truth": original_ground_truth,
        "cache_dir": "/tmp/cache",
        "dataset_name": "THEIA_E3",
    }
    assert call_log == [
        "get_ground_truth",
        "fetch_node_metadata",
        "get_training_nodes_from_csv",
        ("train_word2vec", {"10": ["process", "proc"]}, {"10"}),
        ("gen_edge_fused_tw", {"10": ["process", "proc"]}, {"path": "model"}),
    ]


def test_build_graphs_cache_path_relabels_and_skips_heavy_dependencies(monkeypatch):
    config = make_config()
    cached_graphs = {"train": ["cached"], "val": [], "test": []}
    cached_ground_truth = {"attack": {"nids": [7], "contaminated_nids": [8]}}
    relabel_calls = []

    monkeypatch.setattr(
        "tasks.build_graphs.load_graph_cache",
        lambda cache_dir, dataset_name, cfg: (cached_graphs, cached_ground_truth),
    )
    monkeypatch.setattr(
        "tasks.build_graphs.relabel_graphs",
        lambda graphs, malicious_nodes: relabel_calls.append((graphs, malicious_nodes)),
    )

    def fail(*args, **kwargs):
        raise AssertionError("heavy dependency should not run on cache hit")

    monkeypatch.setattr("tasks.build_graphs.get_ground_truth", fail)
    monkeypatch.setattr("tasks.build_graphs.fetch_node_metadata", fail)
    monkeypatch.setattr("tasks.build_graphs.get_training_nodes_from_csv", fail)
    monkeypatch.setattr("tasks.build_graphs.train_word2vec", fail)
    monkeypatch.setattr("tasks.build_graphs.gen_edge_fused_tw", fail)
    monkeypatch.setattr("tasks.build_graphs.save_graph_cache", fail)

    graphs, filtered_ground_truth = build_graphs(config)

    assert graphs is cached_graphs
    assert filtered_ground_truth == cached_ground_truth
    assert relabel_calls == [(cached_graphs, {7, 8})]


def test_build_graphs_wraps_ground_truth_loading_errors(monkeypatch):
    config = make_config()

    monkeypatch.setattr(
        "tasks.build_graphs.load_graph_cache",
        lambda cache_dir, dataset_name, cfg: (None, None),
    )
    monkeypatch.setattr(
        "tasks.build_graphs.get_ground_truth",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(ValueError, match="Failed to load ground truth: boom"):
        build_graphs(config)
