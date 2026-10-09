from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from openpi.training.cdlam_memory_dataset import classify_memory_segments, drop_execution_lam_tail

ROOT = Path(__file__).resolve().parents[1]


class MemoryContractTest(unittest.TestCase):
    def test_segment_boundary(self) -> None:
        result = classify_memory_segments(np.arange(1, 8, dtype=np.int64), exec_start_idx=4)
        np.testing.assert_array_equal(result, np.asarray([1, 1, 1, 2, 3, 3, 3], dtype=np.int32))

    def test_fixed_lag_only_drops_execution_tail(self) -> None:
        latents = np.arange(15 * 2, dtype=np.float32).reshape(15, 2)
        mask = np.ones(15, dtype=np.bool_)
        segments = np.asarray([1, 1, 2] + [3] * 12, dtype=np.int32)
        kept_latents, kept_mask, kept_segments = drop_execution_lam_tail(latents, mask, segments, 10)
        np.testing.assert_array_equal(kept_latents, latents[:5])
        np.testing.assert_array_equal(kept_mask, np.ones(5, dtype=np.bool_))
        np.testing.assert_array_equal(kept_segments, np.asarray([1, 1, 2, 3, 3], dtype=np.int32))

    def test_public_data_contract(self) -> None:
        contract = json.loads((ROOT / "configs" / "data_contract.json").read_text())
        self.assertNotIn("logical_dataset", contract)
        self.assertEqual(contract["video_dataset"]["episodes"], 16000)
        self.assertEqual(contract["video_dataset"]["frames"], 7636924)
        self.assertEqual(contract["training"]["memory_lag"], 10)
        self.assertEqual(contract["training"]["max_token_len"], 200)
        self.assertFalse(contract["training"]["memory_same_prefix_block"])
        self.assertEqual(contract["memory_cache"]["encoder"], "DreamDojo LAM 400k")
        self.assertEqual(
            contract["memory_cache"]["checkpoint_sha256"],
            "12ce318e48d5790fb3773f1027edd5a3babe8965027b4957e3ed62d5d19fc386",
        )
        self.assertEqual(
            contract["memory_cache"]["latent_layout"],
            "concat(image_z_mu[32],wrist_image_z_mu[32])",
        )
        self.assertEqual(contract["memory_cache"]["latent_dtype"], "float32")
        self.assertEqual(contract["memory_cache"]["views"], ["image", "wrist_image"])

    def test_vendored_dreamdojo_provenance(self) -> None:
        upstream = json.loads((ROOT / "third_party" / "dreamdojo" / "UPSTREAM.json").read_text())
        self.assertEqual(upstream["repository"], "https://github.com/NVIDIA/DreamDojo.git")
        self.assertEqual(upstream["commit"], "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff")
        self.assertTrue((ROOT / "third_party" / "dreamdojo" / "LICENSE").is_file())
        for path in upstream["vendored_paths"]:
            vendored = ROOT / "third_party" / "dreamdojo" / "runtime" / path
            self.assertTrue(vendored.is_file(), path)
            self.assertEqual(hashlib.sha256(vendored.read_bytes()).hexdigest(), upstream["sha256"][path])


if __name__ == "__main__":
    unittest.main()
