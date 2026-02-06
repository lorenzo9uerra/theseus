import os
import re
import time
from datetime import datetime
from functools import wraps

import polars as pl
import psutil
import pytz
import torch
from tqdm import tqdm


def timed_execution(func):
    """Decorator to log execution time and peak memory usage."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        log(f"======= START TASK {func.__name__} =======")
        start = time.time()

        process = psutil.Process()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        result = func(*args, **kwargs)

        peak_mem_gb = process.memory_info().rss / 1024**3

        if torch.cuda.is_available():
            peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            log(
                f"Task {func.__name__} completed in {time.time() - start:.1f}s, peak RAM: {peak_mem_gb:.2f} GB, peak GPU memory: {peak_gpu_mem_gb:.2f} GB"
            )
        else:
            log(
                f"Task {func.__name__} completed in {time.time() - start:.1f}s, peak RAM: {peak_mem_gb:.2f} GB"
            )

        return result

    return wrapper


def datetime_to_ns_time_us(date):
    """Convert 'YYYY-MM-DD HH:MM:SS' (US/Eastern) to nanosecond timestamp."""
    tz = pytz.timezone("US/Eastern")
    timearray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(time.mktime(timearray))
    timestamp = tz.localize(dt).timestamp() * 1_000_000_000
    return int(timestamp)


def tokenize_process(sentence: str, max_tokens: int = 50):
    """
    Tokenize process command lines into unigrams and bigrams.

    Tokenization: whitespace split + consecutive bigrams.
    Example: "/usr/bin/python -u script.py" ->
        [("/usr/bin/python", "command"), ("-u", "command"), ("script.py", "command"),
         ("/usr/bin/python_-u", "command"), ("-u_script.py", "command")]
    """
    if not sentence or sentence.strip() == "":
        return []

    parts = [p for p in sentence.strip().split() if p]
    if not parts:
        return []

    tokens = [(part, "command") for part in parts]
    tokens.extend(
        (f"{parts[i]}_{parts[i + 1]}", "command") for i in range(len(parts) - 1)
    )

    return tokens[:max_tokens]


def tokenize_file(sentence: str):
    """
    Tokenize file paths into directory components, extension, and bigrams.

    Tokenization: split by "/" + file extension + consecutive bigrams.
    Example: "/home/user/docs/file.txt" ->
        [("home", "path"), ("user", "path"), ("docs", "path"), ("file.txt", "path"),
         ("ext_txt", "path"), ("home_user", "path"), ("user_docs", "path"), ("docs_file.txt", "path")]
    """
    if not sentence or sentence.strip() == "":
        return []

    normalized = re.sub(r"\\+", "/", sentence.strip())
    parts = [p for p in normalized.strip("/").split("/") if p]
    if not parts:
        return []

    tokens = [(part, "path") for part in parts]

    # Add file extension as special token
    filename = parts[-1]
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1]
        tokens.append((f"ext_{ext}", "path"))

    tokens.extend((f"{parts[i]}_{parts[i + 1]}", "path") for i in range(len(parts) - 1))

    return tokens


def tokenize_netflow(sentence: str):
    """
    Tokenize network flow into IP/hostname tokens, port tokens, and semantic categories.

    Handles formats:
    - Space-separated: "src_ip src_port dst_ip dst_port" (e.g., "192.168.1.1 80 10.0.0.1 443")
    - IP:port: "192.168.1.1:80"

    Example: "192.168.1.1 80 10.0.0.1 443" ->
        [("ip_192_168_1_1", "netflow"), ("subnet_192_168", "netflow"), ("private_ip_class_c", "netflow"),
         ("port_80", "netflow"), ("system_port", "netflow"), ("src_192.168.1.1_80", "netflow"),
         ("ip_10_0_0_1", "netflow"), ("subnet_10_0", "netflow"), ("private_ip_class_a", "netflow"),
         ("port_443", "netflow"), ("system_port", "netflow"), ("dst_10.0.0.1_443", "netflow")]
    """
    if not sentence or sentence.strip() == "":
        return []

    sentence = sentence.strip()
    tokens = []
    parts = sentence.split()

    if (
        len(parts) >= 4
        and parts[1].lstrip("-").isdigit()
        and parts[3].lstrip("-").isdigit()
    ):
        # Space-separated format: src_addr src_port dst_addr dst_port
        tokens.extend(_tokenize_ip_or_hostname(parts[0]))
        tokens.extend(_tokenize_port(parts[1]))
        tokens.append(f"src_{parts[0]}_{parts[1]}")
        tokens.extend(_tokenize_ip_or_hostname(parts[2]))
        tokens.extend(_tokenize_port(parts[3]))
        tokens.append(f"dst_{parts[2]}_{parts[3]}")
    elif len(parts) == 1 and ":" in parts[0]:
        # Single IP:port format
        ip_part, port_part = parts[0].rsplit(":", 1)
        tokens.extend(_tokenize_ip_or_hostname(ip_part))
        tokens.extend(_tokenize_port(port_part))
    else:
        for part in parts:
            if ":" in part and part.count(":") == 1:
                ip_part, port_part = part.split(":", 1)
                tokens.extend(_tokenize_ip_or_hostname(ip_part))
                tokens.extend(_tokenize_port(port_part))
            else:
                tokens.extend(_tokenize_ip_or_hostname(part))

    return [(token, "netflow") for token in tokens]


def _tokenize_ip(ip_str: str):
    """Tokenize IPv4 into: full IP, subnet prefix, and private/public class."""
    tokens = []
    if "." not in ip_str:
        return tokens

    ip_parts = ip_str.split(".")
    if len(ip_parts) != 4 or not all(p.isdigit() for p in ip_parts):
        return tokens

    tokens.append(f"ip_{'_'.join(ip_parts)}")
    tokens.append(f"subnet_{ip_parts[0]}_{ip_parts[1]}")

    first_octet = int(ip_parts[0])
    if first_octet == 10:
        tokens.append("private_ip_class_a")
    elif 172 <= first_octet <= 175:
        tokens.append("private_ip_class_b")
    elif first_octet == 192:
        tokens.append("private_ip_class_c")
    elif first_octet == 127:
        tokens.append("localhost_ip")
    else:
        tokens.append("public_ip")

    return tokens


def _tokenize_ip_or_hostname(addr_str: str):
    """Route to appropriate tokenizer based on address format."""
    if addr_str in ["localhost", "::1", "0.0.0.0"]:
        return ["localhost"]
    if ":" in addr_str:
        return [f"ipv6_{addr_str.replace(':', '_')}"]
    if "." in addr_str:
        ip_parts = addr_str.split(".")
        if len(ip_parts) == 4 and all(p.isdigit() for p in ip_parts):
            return _tokenize_ip(addr_str)
        return [f"hostname_{addr_str.replace('.', '_')}"]
    return [f"host_{addr_str}"]


def _tokenize_port(port_str: str):
    """Tokenize port into: port number and category (system/user/dynamic)."""
    try:
        port_num = int(port_str)
    except ValueError:
        return [f"port_{port_str}"]

    if port_num == -1:
        return ["port_unknown"]

    tokens = [f"port_{port_str}"]
    if port_num < 1024:
        tokens.append("system_port")
    elif port_num < 49152:
        tokens.append("user_port")
    else:
        tokens.append("dynamic_port")
    return tokens


def tokenize_node_description(node_description, node_type):
    """
    Dispatch to appropriate tokenizer based on node type.

    Returns list of (token, token_type) tuples where token_type in {'path', 'command', 'netflow'}.
    """
    if not node_description:
        return []
    if node_type == "process":
        return tokenize_process(node_description)
    if node_type == "file":
        return tokenize_file(node_description)
    if node_type == "netflow":
        return tokenize_netflow(node_description)
    raise ValueError(f"Invalid node type: {node_type}")


def log(msg: str, return_line=False, pre_return_line=False, *args, **kwargs):
    """Print timestamped log message."""
    if pre_return_line:
        print("")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {msg}", *args, **kwargs)
    if return_line:
        print("")


def log_tqdm(iterator, desc="", miniters=None, logging=True):
    """Wrap iterator with tqdm progress bar (timestamped description)."""
    if not logging:
        return iterator
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return tqdm(iterator, desc=f"{timestamp} - {desc}", miniters=miniters)


def create_one_hot(type_list):
    """Generate one-hot encoding dict: {name: one_hot_tensor}."""
    num_types = len(type_list)
    vectors = torch.nn.functional.one_hot(
        torch.arange(num_types), num_classes=num_types
    ).float()
    return {name: vectors[i] for i, name in enumerate(type_list)}


def read_node_table(data_dir, table_name, columns=None):
    """Read node table, trying Parquet first then CSV."""
    parquet_path = os.path.join(data_dir, f"{table_name}.parquet")
    csv_path = os.path.join(data_dir, f"{table_name}.csv")
    if os.path.exists(parquet_path):
        return pl.read_parquet(parquet_path, columns=columns)
    elif os.path.exists(csv_path):
        return pl.read_csv(csv_path, columns=columns)
    return None
