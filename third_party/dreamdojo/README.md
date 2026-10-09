# Vendored DreamDojo LAM runtime

This directory contains the minimal upstream DreamDojo LAM modules required by
LTM-Pi's cache builder and online inference server. The files under
`runtime/external/lam/modules/` are copied without modification from NVIDIA
DreamDojo commit `02f119b759d5c7f84a399fdeea3c6e82e7ed6cff`.

`runtime/dreamdojo_adapter.py` is an LTM-Pi adapter for deterministic 32-D
posterior-mean extraction and the preprocessing used for the released cache.
The model checkpoint is not redistributed.

Upstream: <https://github.com/NVIDIA/DreamDojo>

License: Apache-2.0; see `LICENSE`.
