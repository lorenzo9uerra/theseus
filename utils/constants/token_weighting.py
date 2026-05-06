"""Token weighting mode constants and validation helpers."""

DEFAULT_TOKEN_WEIGHTING_MODE = "reverted_netflow"

TOKEN_WEIGHTING_MODE_TO_REVERSED_NODE_TYPES = {
    "standard_decay": frozenset(),
    "reverted_netflow": frozenset({"netflow"}),
    "reverted_file": frozenset({"file"}),
    "reverted_netflow_and_file": frozenset({"netflow", "file"}),
}

TOKEN_WEIGHTING_MODE_CACHE_SUFFIX = {
    "standard_decay": "tw_std_decay",
    "reverted_netflow": "tw_rev_netflow",
    "reverted_file": "tw_rev_file",
    "reverted_netflow_and_file": "tw_rev_netflow_file",
}

TOKEN_WEIGHTING_MODES = tuple(TOKEN_WEIGHTING_MODE_TO_REVERSED_NODE_TYPES.keys())


def resolve_token_weighting_mode(mode_value):
    """Normalize and validate a token weighting mode string."""
    if mode_value is None:
        return DEFAULT_TOKEN_WEIGHTING_MODE

    normalized_mode = (
        str(mode_value).strip().lower().replace("-", "_").replace(" ", "_")
    )

    if normalized_mode not in TOKEN_WEIGHTING_MODE_TO_REVERSED_NODE_TYPES:
        valid_modes = ", ".join(sorted(TOKEN_WEIGHTING_MODE_TO_REVERSED_NODE_TYPES))
        raise ValueError(
            f"Invalid token weighting mode '{mode_value}'. Valid modes: {valid_modes}"
        )

    return normalized_mode


def get_token_weighting_mode(config):
    """Resolve token weighting mode from config with backward-compatible defaults."""
    mode_value = getattr(config, "token_weighting_mode", None)
    if mode_value is None and hasattr(config, "word2vec"):
        mode_value = getattr(config.word2vec, "token_weighting_mode", None)
    return resolve_token_weighting_mode(mode_value)
