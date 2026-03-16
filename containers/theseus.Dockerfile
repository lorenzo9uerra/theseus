FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/theseus/.venv \
    PATH=/opt/theseus/.venv/bin:/root/.local/bin:${PATH}

WORKDIR /opt/theseus

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.10.10

COPY LICENSE README.md main.py pyproject.toml uv.lock ./
COPY configs ./configs
COPY ground_truth ./ground_truth
COPY models ./models
COPY scripts ./scripts
COPY tasks ./tasks
COPY utils ./utils

RUN uv sync --frozen --no-dev
RUN .venv/bin/python -c "import torch; import torch_geometric"

CMD ["bash"]
