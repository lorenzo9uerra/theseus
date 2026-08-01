# Container

The Dockerfile provides an optional environment for the main Theseus E3 evaluation. It does not include datasets, caches, checkpoints, or result logs; mount those directories at runtime. Magic, PIDSMaker, and ATLASv2 use their native environments.

## Docker

Build:

```bash
docker build -f containers/theseus.Dockerfile -t theseus:latest .
```

Run the main Theseus E3 paper evaluation with mounted artifacts:

```bash
mkdir -p results_rerun
docker run --rm -it --gpus all \
  -v "$PWD/data:/opt/theseus/data" \
  -v "$PWD/cache:/opt/theseus/cache" \
  -v "$PWD/checkpoints:/opt/theseus/checkpoints" \
  -v "$PWD/results_rerun:/opt/theseus/results_rerun" \
  -e RESULT_DIR=results_rerun \
  theseus:latest \
  bash -lc './scripts/reproduce_results.sh'
```
