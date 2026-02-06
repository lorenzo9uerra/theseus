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
