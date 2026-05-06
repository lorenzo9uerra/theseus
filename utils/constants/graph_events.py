"""
Graph event types and node types for different datasets.

Simple list-based definitions, for PyTorch one-hot encoding.
"""

# === NODE TYPES ===
NODE_TYPES = ["process", "file", "netflow"]

# === DATASET-SPECIFIC EVENT TYPES ===

DARPA_TC_EVENTS = [
    "EVENT_CONNECT",
    "EVENT_EXECUTE",
    "EVENT_OPEN",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_SENDMSG",
    "EVENT_SENDTO",
    "EVENT_WRITE",
    "EVENT_FORK",
]

ATLASV2_EVENTS = [
    "EVENT_CONNECT",
    "EVENT_EXECUTE",
    "EVENT_FORK",
    "EVENT_OPEN",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_WRITE",
]


def get_dataset_event_types(dataset_name: str) -> list[str]:
    name = (dataset_name or "").lower()
    if name.startswith("atlasv2"):
        return ATLASV2_EVENTS
    return DARPA_TC_EVENTS
