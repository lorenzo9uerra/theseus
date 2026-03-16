# Containers

These files provide a publication-oriented runtime for the main Theseus pipeline.

They intentionally exclude large mutable artifacts such as `data/`, `cache/`, `checkpoints/`, and `results/`. Mount those at runtime.

## Recommended Path

The recommended path is:

1. build and publish the Docker image with GitHub Actions to GHCR;
2. pull it on the cluster with Apptainer;
3. bind `data/`, `cache/`, `checkpoints/`, and `results/` at runtime.

The workflow is defined in [publish-theseus-image.yml](../.github/workflows/publish-theseus-image.yml).

## Docker

Build:

```bash
docker build -f containers/theseus.Dockerfile -t theseus:latest .
```

Run the paper evaluation with mounted artifacts:

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

On the Telecom Paris cluster, load the runtime first:

```bash
module load apptainer
```

If the image has already been published to GHCR, pull it directly:

```bash
apptainer pull theseus.sif docker://ghcr.io/<owner>/<repo>:latest
```

You can also pin a commit-specific image:

```bash
apptainer pull theseus.sif docker://ghcr.io/<owner>/<repo>:sha-<commit>
```

Local builds from [theseus.def](./theseus.def) are still possible on systems where unprivileged Apptainer builds are enabled:

```bash
apptainer build theseus.sif containers/theseus.def
```

Run the paper evaluation on a cluster:

```bash
apptainer exec --nv \
  --bind "$PWD/data:/opt/theseus/data" \
  --bind "$PWD/cache:/opt/theseus/cache" \
  --bind "$PWD/checkpoints:/opt/theseus/checkpoints" \
  --bind "$PWD/results:/opt/theseus/results" \
  theseus.sif \
  ./scripts/reproduce_results.sh
```
