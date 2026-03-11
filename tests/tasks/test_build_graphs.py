from types import SimpleNamespace

import numpy as np
import torch

from tasks.build_graphs import create_graph_window


def make_config():
    return SimpleNamespace(
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
        all_malicious_nodes={3},
    )

    assert split == "val"
    assert graph.original_n_id.tolist() == [1, 2, 3]
    assert graph.y.tolist() == [0, 0, 1]
    assert torch.allclose(
        graph.x[:, :2],
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        graph.x[:, 2],
        torch.tensor(np.log1p([1, 2, 1]), dtype=torch.float32),
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
