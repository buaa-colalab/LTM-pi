from __future__ import annotations

import unittest

import numpy as np

from scripts.serve import MemorySession


class _Tokenizer:
    def encode(self, _text: str, add_bos: bool = False) -> list[int]:
        return [1] if add_bos else [2]


class _PromptTokenizer:
    _tokenizer = _Tokenizer()


class _CapturePolicy:
    def infer(self, element, noise):
        valid = element["memory_mask"]
        np.testing.assert_array_equal(element["memory_segment_ids"][valid], [1, 1, 2, 3, 3])
        np.testing.assert_array_equal(element["memory_latents"][valid, 0], [0, 1, 2, 3, 4])
        return {"actions": np.zeros((50, 8), dtype=np.float32)}


class ServerLagTest(unittest.TestCase):
    def test_server_hides_ten_execution_tokens_only(self) -> None:
        session = MemorySession(
            policy=_CapturePolicy(),
            dreamdojo_model=None,
            encode_full=None,
            preprocess=None,
            device="cpu",
            memory_horizon=1800,
            latent_dim=64,
            action_horizon=50,
            action_dim=8,
            prompt_tokenizer=_PromptTokenizer(),
            max_token_len=200,
            pair_batch_size=16,
            use_memory_segments=True,
            use_demo_anchor=True,
            use_execution_anchor=True,
            transition_frame_stride=1,
            two_view_memory=True,
            memory_lag=10,
            latent_storage_dtype="float32",
        )
        session.latents = [np.full(64, index, dtype=np.float32) for index in range(15)]
        session.segment_ids = [1, 1, 2] + [3] * 12
        session.total_frames = 16
        session.total_transitions = 15
        session.exec_start_idx = 3
        session.demo_start_image = np.zeros((2, 2, 3), dtype=np.uint8)
        session.execution_start_image = np.zeros((2, 2, 3), dtype=np.uint8)

        result = session.infer({"prompt": "test"})

        self.assertEqual(result["memory_timing"]["stored_memory_length"], 15)
        self.assertEqual(result["memory_timing"]["memory_length"], 5)
        self.assertEqual(result["memory_timing"]["lag_hidden_transitions"], 10)


if __name__ == "__main__":
    unittest.main()
