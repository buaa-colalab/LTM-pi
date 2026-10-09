# Changelog

## Unreleased

- Allow Hatch direct-reference metadata and constrain `datasets` to the
  LeRobot-compatible pre-4.x API.
- Write generated normalization statistics directly to the configured
  `ROBOMME_ASSETS_ROOT` used by training preflight.

## 1.1.0 - 2026-10-06

- Rename the standalone release to `ltm_pi_robomme`.
- Replace the release encoder with DreamDojo LAM 400k and pin its checkpoint checksum.
- Vendor the minimal DreamDojo LAM runtime and upstream Apache-2.0 license.
- Document separate anchor-cache, memory-cache, and norm-stat generation commands.
- Use the H.264 video dataset as the only dataset input, including direct
  head/wrist decoding for anchor and DreamDojo memory-cache generation.

## 1.0.0 - 2026-10-06

- Release the self-contained OpenPI training and inference source tree.
- Add the RoboMME 16×1000 H.264 fixed-lag-10 configuration.
- Match online inference to training: 200 prompt tokens, LMV causal masking,
  FP32 two-view memory features, and a 10-token execution-memory lag.
- Add data/config/checkpoint validation, a minimal WebSocket client, tests,
  reproducible dependency pins, and CI.
