# Containers

These recipes provide a publication-oriented runtime for the main Theseus pipeline.

They intentionally exclude large mutable artifacts such as `data/`, `cache/`, `checkpoints/`, and `results/`. Mount those at runtime.

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

Build:

```bash
apptainer build theseus.sif containers/theseus.def
```

If your cluster does not allow local unprivileged builds, build the image on a machine with Docker/Apptainer support or use a remote Singularity build service, then copy the resulting `.sif` to the cluster.

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
