import json
import os
import pickle as pkl

import dgl
import networkx as nx
import torch.nn.functional as F

# Resolve MAGIC root directory (parent of utils/)
_MAGIC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_MAGIC_ROOT, "data")


def transform_graph(g, node_feature_dim, edge_feature_dim):
    new_g = g.clone()
    new_g.ndata["attr"] = F.one_hot(
        g.ndata["type"].view(-1), num_classes=node_feature_dim
    ).float()
    new_g.edata["attr"] = F.one_hot(
        g.edata["type"].view(-1), num_classes=edge_feature_dim
    ).float()
    return new_g


def preload_entity_level_dataset(path):
    path = os.path.join(_DATA_DIR, path)
    if os.path.exists(path + "/metadata.json"):
        pass
    else:
        print("transforming")
        train_gs = [
            dgl.from_networkx(
                nx.node_link_graph(g), node_attrs=["type"], edge_attrs=["type"]
            )
            for g in pkl.load(open(path + "/train.pkl", "rb"))
        ]
        print("transforming")
        test_gs = [
            dgl.from_networkx(
                nx.node_link_graph(g), node_attrs=["type"], edge_attrs=["type"]
            )
            for g in pkl.load(open(path + "/test.pkl", "rb"))
        ]
        malicious = pkl.load(open(path + "/malicious.pkl", "rb"))

        node_feature_dim = 0
        for g in train_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        for g in test_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        node_feature_dim += 1
        edge_feature_dim = 0
        for g in train_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        for g in test_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        edge_feature_dim += 1
        result_test_gs = []
        for g in test_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)
            result_test_gs.append(g)
        result_train_gs = []
        for g in train_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)
            result_train_gs.append(g)
        metadata = {
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "malicious": malicious,
            "n_train": len(result_train_gs),
            "n_test": len(result_test_gs),
        }
        with open(path + "/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        for i, g in enumerate(result_train_gs):
            with open(path + f"/train{i}.pkl", "wb") as f:
                pkl.dump(g, f)
        for i, g in enumerate(result_test_gs):
            with open(path + f"/test{i}.pkl", "wb") as f:
                pkl.dump(g, f)


def load_metadata(path):
    preload_entity_level_dataset(path)
    with open(os.path.join(_DATA_DIR, path, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


def load_entity_level_dataset(path, t, n):
    preload_entity_level_dataset(path)
    with open(os.path.join(_DATA_DIR, path, f"{t}{n}.pkl"), "rb") as f:
        data = pkl.load(f)
    return data
