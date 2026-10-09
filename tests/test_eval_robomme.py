from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts import eval_robomme


class EvalDatabaseTest(unittest.TestCase):
    def test_queue_retry_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "jobs.sqlite3"
            eval_robomme.init_db(db_path, max_attempts=3, run_name="unit-test")

            with sqlite3.connect(db_path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 800)

            with eval_robomme.connect_db(db_path) as db:
                job = eval_robomme.claim_job(
                    db,
                    worker_id="g0w0",
                    gpu_id=0,
                    stale_seconds=1800,
                    max_attempts=3,
                )
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual((job["task"], job["episode"], job["attempts"]), ("BinFill", 0, 1))

            eval_robomme.retry_or_fail_job(
                db_path,
                "BinFill",
                0,
                worker_id="g0w0",
                error="synthetic retry",
                max_attempts=3,
            )
            with eval_robomme.connect_db(db_path) as db:
                retried = db.execute(
                    "SELECT status, attempts FROM jobs WHERE task='BinFill' AND episode=0"
                ).fetchone()
                self.assertEqual((retried["status"], retried["attempts"]), ("pending", 1))
                job = eval_robomme.claim_job(
                    db,
                    worker_id="g0w0",
                    gpu_id=0,
                    stale_seconds=1800,
                    max_attempts=3,
                )
            assert job is not None
            self.assertEqual((job["task"], job["episode"], job["attempts"]), ("BinFill", 0, 2))

            eval_robomme.finish_job(
                db_path,
                "BinFill",
                0,
                "g0w0",
                {
                    "status": "success",
                    "outcome": "success",
                    "duration_s": 1.0,
                    "steps": 10,
                    "model_seed": 42,
                    "task_goal": "test",
                    "difficulty": "test",
                    "server_metadata": {},
                    "memory_stats": {},
                },
            )
            totals = eval_robomme.build_summary(db_path)["totals"]
            self.assertEqual(totals["success"], 1)
            self.assertEqual(totals["pending"], 799)
            self.assertEqual(totals["running"], 0)


if __name__ == "__main__":
    unittest.main()
