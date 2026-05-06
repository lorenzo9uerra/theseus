# Containers

These files provide an optional publication-oriented runtime for the main Theseus
E3 evaluation pipeline.

They intentionally exclude large mutable artifacts such as `data/`, `cache/`, `checkpoints/`, and `results/`. Mount those at runtime.

This container path is intentionally scoped to the main `Theseus` E3 workflow
driven by `scripts/reproduce_results.sh`. It does not currently aim to cover:

- the `MAGIC` baseline;
- the `Orthrus` / `Velox` `PIDSMaker` baseline stack;
- the full ATLASv2 eval-only release path in `scripts/reproduce_atlasv2_results.sh`.

Those paths remain on the native `uv.lock` workflow described in the repository
root. The limiting issue is that the baseline stack currently depends on a
different Python / CUDA / PyTorch toolchain than the main Theseus runtime, so a
single lightweight image would be misleading.

## Recommended Path

The recommended container path is:

1. build and publish the Docker image with GitHub Actions to GHCR;
2. pull it on the cluster with Apptainer;
3. bind `data/`, `cache/`, `checkpoints/`, and `results/` at runtime.

The workflow is defined in [publish-theseus-image.yml](../.github/workflows/publish-theseus-image.yml). The native `uv.lock` workflow documented in the repository root remains the canonical reproduction path; the container is provided as an additional convenience for evaluators running the main Theseus E3 path.

Published images follow the GHCR pattern:

```text
ghcr.io/lorenzo9uerra/theseus:<tag>
```

In practice:

- use `:latest` only for the default branch;
- use `:docker` while validating the container branch;
- use `:sha-<commit>` for artifact evaluation and archival reproduction.

## Docker

Build:

```bash
docker build -f containers/theseus.Dockerfile -t theseus:latest .
```

Run the main Theseus E3 paper evaluation with mounted artifacts:

```bash
docker run --rm -it --gpus all \
  -v "$PWD/data:/opt/theseus/data" \
  -v "$PWD/cache:/opt/theseus/cache" \
  -v "$PWD/checkpoints:/opt/theseus/checkpoints" \
  -v "$PWD/results:/opt/theseus/results" \
  theseus:latest \
  bash -lc './scripts/reproduce_results.sh'
```

## Apptainer

On clusters, load the Apptainer runtime first:

```bash
module load apptainer
```

If the image has already been published to GHCR, pull it directly:

```bash
apptainer pull theseus.sif docker://ghcr.io/lorenzo9uerra/theseus:latest
```

For a pinned, reproducible image, pull a specific published commit tag:

```bash
apptainer pull theseus.sif docker://ghcr.io/lorenzo9uerra/theseus:sha-<commit>
```

The same naming works with Docker:

```bash
docker pull ghcr.io/lorenzo9uerra/theseus:sha-<commit>
```

Run the main Theseus E3 paper evaluation on a cluster:

```bash
apptainer exec --nv \
  --bind "$PWD/data:/opt/theseus/data" \
  --bind "$PWD/cache:/opt/theseus/cache" \
  --bind "$PWD/checkpoints:/opt/theseus/checkpoints" \
  --bind "$PWD/results:/opt/theseus/results" \
  theseus.sif \
  ./scripts/reproduce_results.sh
```
