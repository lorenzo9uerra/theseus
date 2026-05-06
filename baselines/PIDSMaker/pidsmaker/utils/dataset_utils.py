# The directions of the following edge types need to be reversed
edge_reversed = [
    "EVENT_EXECUTE",
    "EVENT_LSEEK",
    "EVENT_MMAP",
    "EVENT_OPEN",
    "EVENT_ACCEPT",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_READ_SOCKET_PARAMS",
    "EVENT_CHECK_FILE_ATTRIBUTES",
    "READ",
]

# The following edges are not considered to construct the
# temporal graph for experiments.
exclude_edge_type = set(
    [
        "EVENT_FCNTL",  # EVENT_FCNTL does not have any predicate
        "EVENT_OTHER",  # EVENT_OTHER does not have any predicate
        "EVENT_ADD_OBJECT_ATTRIBUTE",  # This is used to add attributes to an object that was incomplete at the time of publish
        "EVENT_FLOWS_TO",  # No corresponding system call event
    ]
)

rel2id_darpa_tc = {
    1: "EVENT_CONNECT",
    "EVENT_CONNECT": 1,
    2: "EVENT_EXECUTE",
    "EVENT_EXECUTE": 2,
    3: "EVENT_OPEN",
    "EVENT_OPEN": 3,
    4: "EVENT_READ",
    "EVENT_READ": 4,
    5: "EVENT_RECVFROM",
    "EVENT_RECVFROM": 5,
    6: "EVENT_RECVMSG",
    "EVENT_RECVMSG": 6,
    7: "EVENT_SENDMSG",
    "EVENT_SENDMSG": 7,
    8: "EVENT_SENDTO",
    "EVENT_SENDTO": 8,
    9: "EVENT_WRITE",
    "EVENT_WRITE": 9,
    10: "EVENT_CLONE",
    "EVENT_CLONE": 10,
}
possible_events = {
    ("subject", "subject"): [
        "EVENT_READ",
        "EVENT_WRITE",
        "EVENT_OPEN",
        "EVENT_CONNECT",
        "EVENT_RECVFROM",
        "EVENT_SENDTO",
        "EVENT_CLONE",
        "EVENT_SENDMSG",
        "EVENT_RECVMSG",
    ],
    ("subject", "file"): [
        "EVENT_WRITE",
        "EVENT_CONNECT",
        "EVENT_SENDMSG",
        "EVENT_SENDTO",
        "EVENT_CLONE",
    ],
    ("subject", "netflow"): [
        "EVENT_WRITE",
        "EVENT_SENDTO",
        "EVENT_CONNECT",
        "EVENT_SENDMSG",
    ],
    ("file", "subject"): [
        "EVENT_READ",
        "EVENT_OPEN",
        "EVENT_RECVFROM",
        "EVENT_EXECUTE",
        "EVENT_RECVMSG",
    ],
    ("netflow", "subject"): [
        "EVENT_OPEN",
        "EVENT_READ",
        "EVENT_RECVFROM",
        "EVENT_RECVMSG",
    ],
}
# TODO: do the same for optc (different edges)

rel2id_optc = {
    1: "OPEN",
    "OPEN": 1,
    2: "READ",
    "READ": 2,
    3: "CREATE",
    "CREATE": 3,
    4: "MESSAGE",
    "MESSAGE": 4,
    5: "MODIFY",
    "MODIFY": 5,
    6: "START",
    "START": 6,
    7: "RENAME",
    "RENAME": 7,
    8: "DELETE",
    "DELETE": 8,
    9: "TERMINATE",
    "TERMINATE": 9,
    10: "WRITE",
    "WRITE": 10,
}

# ATLASv2 uses the same normalized EVENT_* vocabulary as the processed Theseus data.
# We keep a dedicated mapping here because the dataset is supplementary and may diverge again,
# but it should reflect the actual Parquet operation values, not the raw ACTION_* taxonomy.
rel2id_atlasv2 = {
    1: "EVENT_CONNECT",
    "EVENT_CONNECT": 1,
    2: "EVENT_EXECUTE",
    "EVENT_EXECUTE": 2,
    3: "EVENT_FORK",
    "EVENT_FORK": 3,
    4: "EVENT_OPEN",
    "EVENT_OPEN": 4,
    5: "EVENT_READ",
    "EVENT_READ": 5,
    6: "EVENT_RECVFROM",
    "EVENT_RECVFROM": 6,
    7: "EVENT_WRITE",
    "EVENT_WRITE": 7,
}


def decrement_dict(d):
    return {
        k - 1 if isinstance(k, int) else k: v - 1 if isinstance(v, int) else v for k, v in d.items()
    }


def get_rel2id(cfg, from_zero=False):
    if cfg.dataset.name in OPTC_DATASETS:
        return decrement_dict(rel2id_optc) if from_zero else rel2id_optc
    elif cfg.dataset.name in ATLASv2_DATASETS:
        return rel2id_atlasv2
    else:
        return decrement_dict(rel2id_darpa_tc) if from_zero else rel2id_darpa_tc


def get_node_map(from_zero=False):
    if from_zero:
        return decrement_dict(ntype2id)
    return ntype2id


def get_num_edge_type(cfg):
    if (
        cfg.dataset.name not in OPTC_DATASETS
        and "edge_type_triplet" in cfg.detection.graph_preprocessing.edge_features
    ):
        return sum([len(events) for events in possible_events.values()])
    return cfg.dataset.num_edge_types


def get_rel2id_considering_triplets(cfg):
    if "edge_type_triplet" in cfg.detection.graph_preprocessing.edge_features:
        return {
            i + 1: e
            for i, e in enumerate(
                [event for events in possible_events.values() for event in events]
            )
        }
    return get_rel2id(cfg)


ntype2id = {
    1: "subject",
    "subject": 1,
    2: "file",
    "file": 2,
    3: "netflow",
    "netflow": 3,
}

OPTC_DATASETS = {"optc_h201", "optc_h501", "optc_h051"}
ATLASv2_DATASETS = {"atlasv2_h1", "atlasv2_h2"}

OPTC_hostname_map = {
    "optc_h051": "SysClient0051",
    "optc_h201": "SysClient0201",
    "optc_h501": "SysClient0501",
}
