from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

import numpy as np

from scripts.precompute_dreamdojo_memory import EpisodeSource, PreparedEpisode, iter_prepared_episodes


def source(index: int) -> EpisodeSource:
    return EpisodeSource(
        episode_index=index,
        episode_chunk=0,
        expected_length=1,
        path=Path(f"episode_{index}.parquet"),
        relative_path=f"episode_{index}.parquet",
    )


def prepared(item: EpisodeSource) -> PreparedEpisode:
    return PreparedEpisode(
        source=item,
        episode={},
        lam_frames=np.zeros((1, 240, 320, 3), dtype=np.uint8),
        read_seconds=0.0,
        crop_seconds=0.0,
        preprocess_seconds=0.0,
    )


class EpisodePrefetchTest(unittest.TestCase):
    def test_prefetch_preserves_order_and_stays_one_episode_ahead(self) -> None:
        items = [source(index) for index in range(3)]
        started: list[int] = []
        second_started = threading.Event()

        def prepare(item: EpisodeSource) -> PreparedEpisode:
            started.append(item.episode_index)
            if item.episode_index == 1:
                second_started.set()
            return prepared(item)

        iterator = iter_prepared_episodes(items, prepare, prefetch_episodes=1)
        first = next(iterator)
        self.assertTrue(second_started.wait(timeout=1.0))
        self.assertEqual(started, [0, 1])
        remainder = list(iterator)
        self.assertEqual([first.source.episode_index, *(item.source.episode_index for item in remainder)], [0, 1, 2])

    def test_disabled_prefetch_is_synchronous_and_equivalent(self) -> None:
        items = [source(index) for index in range(3)]
        synchronous = list(iter_prepared_episodes(items, prepared, prefetch_episodes=0))
        prefetched = list(iter_prepared_episodes(items, prepared, prefetch_episodes=1))
        self.assertEqual(
            [item.source.episode_index for item in synchronous],
            [item.source.episode_index for item in prefetched],
        )

    def test_two_episode_prefetch_preserves_order(self) -> None:
        items = [source(index) for index in range(6)]
        result = list(iter_prepared_episodes(items, prepared, prefetch_episodes=2))
        self.assertEqual([item.source.episode_index for item in result], list(range(6)))

    def test_ten_episode_prefetch_preserves_order(self) -> None:
        items = [source(index) for index in range(12)]
        active = 0
        max_active = 0
        lock = threading.Lock()

        def prepare_with_concurrency_measurement(item: EpisodeSource) -> PreparedEpisode:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return prepared(item)

        result = list(
            iter_prepared_episodes(items, prepare_with_concurrency_measurement, prefetch_episodes=10)
        )
        self.assertEqual([item.source.episode_index for item in result], list(range(12)))
        self.assertEqual(max_active, 1)


if __name__ == "__main__":
    unittest.main()
