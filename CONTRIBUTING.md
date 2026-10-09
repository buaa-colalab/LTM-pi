# Contributing

Use Python 3.11 and keep training and online inference semantics aligned. In
particular, changes to memory ordering, segment IDs, anchors, storage dtype,
prompt length, or lag must update the data contract, tests, README, and runtime
validation together.

Before opening a pull request, run:

```bash
make check
```

For changes involving real data or checkpoints, configure `.env` and also run:

```bash
bin/validate.sh
bin/validate.sh --checkpoint /path/to/committed/checkpoint/STEP
```

Do not commit datasets, model weights, checkpoints, credentials, W&B files, or
machine-specific `.env` files.
