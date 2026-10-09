# Inference protocol

The server keeps an independent causal memory history for every WebSocket
connection. Reuse a connection to continue an episode; send `reset` or open a
new connection to start another episode.

Recommended sequence:

1. Connect and read server metadata.
2. Send `reset`.
3. Send the demonstration and first execution frame with `add_buffer`.
4. Send the current observation for inference.
5. Append new frames and replan as needed.

The first buffer request must include the execution boundary:

```python
{
    "add_buffer": True,
    "images": head_frames[:, None],          # [T, 1, H, W, 3] uint8
    "wrist_images": wrist_frames[:, None],  # [T, 1, H, W, 3] uint8
    "exec_start_idx": 120,
}
```

Subsequent appends may omit `exec_start_idx`; passing `0` means the new block is
entirely execution. The original global boundary remains unchanged.

Inference request:

```python
{
    "observation/image": head_rgb,          # [H, W, 3] uint8
    "observation/wrist_image": wrist_rgb,  # [H, W, 3] uint8
    "observation/state": state,             # [8] float32
    "prompt": "task instruction",
}
```

The response contains `actions` with shape `[50, 8]` and `memory_timing`.
`memory_timing` reports stored memory, visible memory, lag, and the number of
execution tokens hidden for the current request.

The runtime enforces the release contract: fixed lag 10, max token length 200,
FP32 two-view memory, and `memory_same_prefix_block=False`.
