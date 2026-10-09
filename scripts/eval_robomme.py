#!/usr/bin/env python3
"""Resumable RoboMME batch evaluator for the release policy server.

The evaluator uses both head and wrist views for memory and policy calls. The
8-D simulator state is sent only so the server can restore action deltas; the
release model does not tokenize it when ``discrete_state_input=False``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import traceback
from typing import Any

# Must be set before importing SAPIEN/ManiSkill.
os.environ.setdefault("SAPIEN_RENDER_DEVICE", "cuda:0")

import numpy as np


TASKS = (
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)
TERMINAL = ("success", "failure", "error")
SUITES = {
    "Counting": TASKS[0:4],
    "Permanence": TASKS[4:8],
    "Reference": TASKS[8:12],
    "Imitation": TASKS[12:16],
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def connect_db(path: Path, *, initialize_wal: bool = False) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120.0)
    connection.row_factory = sqlite3.Row
    # Changing the journal mode takes an exclusive lock.  Doing it from every
    # worker is both unnecessary and prone to SQLITE_PROTOCOL on an NFS-backed
    # database when many workers connect at once.  Set it once, before workers
    # are launched, and let all subsequent connections inherit the file mode.
    if initialize_wal:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=120000")
    return connection


def init_db(path: Path, *, max_attempts: int, run_name: str) -> None:
    with connect_db(path, initialize_wal=True) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                task TEXT NOT NULL,
                task_order INTEGER NOT NULL,
                episode INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                worker_id TEXT,
                gpu_id INTEGER,
                started_at REAL,
                heartbeat_at REAL,
                finished_at REAL,
                duration_s REAL,
                steps INTEGER,
                model_seed INTEGER,
                outcome TEXT,
                task_goal TEXT,
                difficulty TEXT,
                error TEXT,
                video_path TEXT,
                server_metadata TEXT,
                memory_stats TEXT,
                PRIMARY KEY (task, episode)
            );
            CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, episode, task_order);
            """
        )
        now = time.time()
        run_id = hashlib.sha256(f"{path.resolve()}:{run_name}".encode()).hexdigest()[:16]
        metadata = {
            "schema_version": "1",
            "created_at": str(now),
            "created_at_iso": utc_now(),
            "run_name": run_name,
            "wandb_run_id": run_id,
            "max_attempts": str(max_attempts),
            "expected_jobs": "800",
        }
        for key, value in metadata.items():
            db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, value))
        for task_order, task in enumerate(TASKS):
            for episode in range(50):
                db.execute(
                    "INSERT OR IGNORE INTO jobs(task, task_order, episode, status) VALUES (?, ?, ?, 'pending')",
                    (task, task_order, episode),
                )
        total = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if total != 800:
            raise RuntimeError(f"Evaluation database contains {total} jobs, expected 800")


def meta_dict(db: sqlite3.Connection) -> dict[str, str]:
    return {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM meta")}


def claim_job(
    db: sqlite3.Connection,
    *,
    worker_id: str,
    gpu_id: int,
    stale_seconds: int,
    max_attempts: int,
) -> sqlite3.Row | None:
    now = time.time()
    db.execute("BEGIN IMMEDIATE")
    db.execute(
        """
        UPDATE jobs
        SET status='pending', worker_id=NULL, gpu_id=NULL, started_at=NULL, heartbeat_at=NULL,
            error=COALESCE(error, '') || '\n[reclaimed stale worker]'
        WHERE status='running' AND heartbeat_at < ? AND attempts < ?
        """,
        (now - stale_seconds, max_attempts),
    )
    db.execute(
        """
        UPDATE jobs SET status='error', finished_at=?, outcome='worker_lost',
            error=COALESCE(error, '') || '\n[stale after maximum attempts]'
        WHERE status='running' AND heartbeat_at < ? AND attempts >= ?
        """,
        (now, now - stale_seconds, max_attempts),
    )
    row = db.execute(
        """
        SELECT * FROM jobs WHERE status='pending' AND attempts < ?
        ORDER BY episode ASC, task_order ASC LIMIT 1
        """,
        (max_attempts,),
    ).fetchone()
    if row is None:
        db.commit()
        return None
    db.execute(
        """
        UPDATE jobs SET status='running', attempts=attempts+1, worker_id=?, gpu_id=?,
            started_at=?, heartbeat_at=?, finished_at=NULL, error=NULL
        WHERE task=? AND episode=? AND status='pending'
        """,
        (worker_id, gpu_id, now, now, row["task"], row["episode"]),
    )
    claimed = db.execute(
        "SELECT * FROM jobs WHERE task=? AND episode=?", (row["task"], row["episode"])
    ).fetchone()
    db.commit()
    return claimed


def heartbeat(db_path: Path, task: str, episode: int, worker_id: str) -> None:
    with connect_db(db_path) as db:
        db.execute(
            "UPDATE jobs SET heartbeat_at=? WHERE task=? AND episode=? AND worker_id=? AND status='running'",
            (time.time(), task, episode, worker_id),
        )


def finish_job(db_path: Path, task: str, episode: int, worker_id: str, result: dict[str, Any]) -> None:
    status = result["status"]
    if status not in TERMINAL:
        raise ValueError(f"Invalid terminal status {status!r}")
    with connect_db(db_path) as db:
        cursor = db.execute(
            """
            UPDATE jobs SET status=?, heartbeat_at=?, finished_at=?, duration_s=?, steps=?, model_seed=?,
                outcome=?, task_goal=?, difficulty=?, error=?, video_path=?, server_metadata=?, memory_stats=?
            WHERE task=? AND episode=? AND worker_id=? AND status='running'
            """,
            (
                status,
                time.time(),
                time.time(),
                result.get("duration_s"),
                result.get("steps"),
                result.get("model_seed"),
                result.get("outcome"),
                result.get("task_goal"),
                result.get("difficulty"),
                result.get("error"),
                result.get("video_path"),
                json.dumps(result.get("server_metadata"), sort_keys=True),
                json.dumps(result.get("memory_stats"), sort_keys=True),
                task,
                episode,
                worker_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Worker {worker_id} no longer owns {task} episode {episode}")


def retry_or_fail_job(
    db_path: Path,
    task: str,
    episode: int,
    *,
    worker_id: str,
    error: str,
    max_attempts: int,
) -> None:
    with connect_db(db_path) as db:
        row = db.execute(
            "SELECT attempts FROM jobs WHERE task=? AND episode=? AND worker_id=? AND status='running'",
            (task, episode, worker_id),
        ).fetchone()
        if row is None:
            return
        terminal = int(row["attempts"]) >= max_attempts
        db.execute(
            """
            UPDATE jobs SET status=?, heartbeat_at=?, finished_at=?, outcome=?, error=?
            WHERE task=? AND episode=? AND worker_id=? AND status='running'
            """,
            (
                "error" if terminal else "pending",
                time.time(),
                time.time() if terminal else None,
                "exception" if terminal else "retry",
                error[-20000:],
                task,
                episode,
                worker_id,
            ),
        )


class PolicyClient:
    def __init__(self, host: str, port: int) -> None:
        import websockets.sync.client
        from openpi_client import msgpack_numpy

        self.packer = msgpack_numpy.Packer()
        self.connection = websockets.sync.client.connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            ping_timeout=600,
            open_timeout=120,
            close_timeout=30,
        )
        self.metadata = msgpack_numpy.unpackb(self.connection.recv())
        if not isinstance(self.metadata, dict):
            raise RuntimeError(f"Policy server returned invalid metadata: {type(self.metadata).__name__}")

    def call(self, message: dict[str, Any]) -> dict[str, Any]:
        from openpi_client import msgpack_numpy

        self.connection.send(self.packer.pack(message))
        raw = self.connection.recv()
        if isinstance(raw, str):
            raise RuntimeError(raw)
        response = msgpack_numpy.unpackb(raw)
        if not isinstance(response, dict):
            raise RuntimeError(f"Policy server returned invalid response: {type(response).__name__}")
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.connection.close()


def current_observation(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(obs["front_rgb_list"][-1], dtype=np.uint8),
        np.asarray(obs["wrist_rgb_list"][-1], dtype=np.uint8),
    )


def current_action_transform_state(obs: dict[str, Any]) -> np.ndarray:
    """Return the current raw joints used only to restore delta actions.

    With ``discrete_state_input=False``, PI0.5 does not receive this value as
    a state token. RoboMME nevertheless trains the first seven action values
    as joint deltas, so ``AbsoluteActions`` must add these joints back after
    sampling. The eighth (gripper) coordinate remains absolute.
    """
    joints = np.asarray(obs["joint_state_list"][-1], dtype=np.float32).reshape(-1)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise ValueError(f"Expected finite current joint_state_list[-1] with shape (7,), got {joints.shape}")
    state = np.zeros(8, dtype=np.float32)
    state[:7] = joints
    return state


def append_video_frame(frames: list[np.ndarray], front: np.ndarray, wrist: np.ndarray, *, border: bool = False) -> None:
    import cv2

    frame = np.concatenate((front, wrist), axis=1).astype(np.uint8, copy=False)
    if border:
        frame = frame.copy()
        cv2.rectangle(frame, (1, 1), (frame.shape[1] - 2, frame.shape[0] - 2), (255, 0, 0), 8)
    frames.append(frame)


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".mp4", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        imageio.mimwrite(
            temporary,
            frames,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=16,
            ffmpeg_params=["-preset", "veryfast"],
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def episode_seed(model_seed: int, task_order: int, episode: int) -> int:
    return int(np.random.SeedSequence([model_seed, task_order, episode]).generate_state(1, dtype=np.uint32)[0])


def run_episode(args: argparse.Namespace, job: sqlite3.Row) -> dict[str, Any]:
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    import robomme.robomme_env  # noqa: F401 - register environments

    task = str(job["task"])
    episode = int(job["episode"])
    seed = episode_seed(args.model_seed, int(job["task_order"]), episode)
    started = time.monotonic()
    env = None
    client = None
    frames: list[np.ndarray] = []
    last_heartbeat = time.monotonic()
    memory_stats: dict[str, Any] = {}
    try:
        client = PolicyClient(args.server_host, args.server_port)
        reset_response = client.call({"reset": True, "seed": seed})
        if not reset_response.get("reset_finished"):
            raise RuntimeError(f"Policy reset failed: {reset_response}")

        builder = BenchmarkEnvBuilder(
            env_id=task,
            dataset="test",
            action_space="joint_angle",
            gui_render=False,
            max_steps=args.max_steps,
        )
        if builder.get_episode_num() != 50:
            raise RuntimeError(f"{task} exposes {builder.get_episode_num()} test episodes, expected 50")
        env = builder.make_env_for_episode(episode)
        obs, info = env.reset()
        task_goals = info.get("task_goal")
        task_goal = str(task_goals[0] if isinstance(task_goals, list) else task_goals)
        if not task_goal:
            raise ValueError("Environment returned an empty task goal")
        difficulty = str(getattr(env.unwrapped, "difficulty", info.get("difficulty", "unknown")))

        front_buffer = [np.asarray(image, dtype=np.uint8) for image in obs["front_rgb_list"]]
        wrist_buffer = [np.asarray(image, dtype=np.uint8) for image in obs["wrist_rgb_list"]]
        initial_frames = len(front_buffer)
        if len(front_buffer) != len(wrist_buffer) or not front_buffer:
            raise ValueError("Initial demonstration buffers are empty or misaligned")
        if args.save_video != "none":
            wrists = [np.asarray(image, dtype=np.uint8) for image in obs["wrist_rgb_list"]]
            for index, (front, wrist) in enumerate(zip(front_buffer, wrists, strict=True)):
                if index % args.video_stride == 0:
                    append_video_frame(frames, front, wrist, border=index < len(front_buffer) - 1)

        add_response = client.call(
            {
                "add_buffer": True,
                "images": np.stack(front_buffer, axis=0)[:, None],
                "wrist_images": np.stack(wrist_buffer, axis=0)[:, None],
                "exec_start_idx": len(front_buffer) - 1,
            }
        )
        if not add_response.get("add_buffer_finished"):
            raise RuntimeError(f"Initial add_buffer failed: {add_response}")
        memory_stats = dict(add_response)

        front, wrist = current_observation(obs)
        action_transform_state = current_action_transform_state(obs)
        action_plan: collections.deque[np.ndarray] = collections.deque()
        pending_front: list[np.ndarray] = []
        pending_wrist: list[np.ndarray] = []
        steps = 0
        outcome = "timeout"
        while steps < args.max_steps:
            if not action_plan:
                if pending_front:
                    add_response = client.call(
                        {
                            "add_buffer": True,
                            "images": np.stack(pending_front, axis=0)[:, None],
                            "wrist_images": np.stack(pending_wrist, axis=0)[:, None],
                            "exec_start_idx": 0,
                        }
                    )
                    if not add_response.get("add_buffer_finished"):
                        raise RuntimeError(f"add_buffer failed: {add_response}")
                    memory_stats = dict(add_response)
                    pending_front.clear()
                    pending_wrist.clear()
                output = client.call(
                    {
                        "observation/image": front,
                        "observation/wrist_image": wrist,
                        # Not tokenized with discrete_state_input=False; used
                        # solely by the output transform to restore the first
                        # seven joint deltas to absolute actions.
                        "observation/state": action_transform_state,
                        "prompt": task_goal,
                    }
                )
                actions = np.asarray(output.get("actions"), dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) < args.replan_steps:
                    raise ValueError(f"Policy returned actions with shape {actions.shape}")
                if not np.all(np.isfinite(actions[: args.replan_steps])):
                    raise ValueError("Policy returned non-finite actions")
                action_plan.extend(actions[: args.replan_steps])
                memory_stats = dict(output.get("memory_timing", memory_stats))

            action = action_plan.popleft()
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            status = str((info or {}).get("status", "unknown"))
            if status == "error":
                raise RuntimeError(str((info or {}).get("error_message", "environment status=error")))
            front, wrist = current_observation(obs)
            action_transform_state = current_action_transform_state(obs)
            pending_front.append(front.copy())
            pending_wrist.append(wrist.copy())
            if args.save_video != "none" and steps % args.video_stride == 0:
                append_video_frame(frames, front, wrist)
            if time.monotonic() - last_heartbeat >= args.heartbeat_seconds:
                heartbeat(args.db, task, episode, args.worker_id)
                last_heartbeat = time.monotonic()
            if terminated or truncated:
                outcome = status
                break

        terminal_status = "success" if outcome == "success" else "failure"
        memory_stats["initial_frames"] = initial_frames
        video_path: str | None = None
        if args.save_video == "all" or (args.save_video == "failures" and terminal_status != "success"):
            safe_goal = hashlib.sha256(task_goal.encode()).hexdigest()[:8]
            path = args.results_dir / "videos" / f"{task}_ep{episode:02d}_{outcome}_{safe_goal}.mp4"
            save_video(path, frames, max(1, args.video_fps // args.video_stride))
            video_path = str(path)
        return {
            "status": terminal_status,
            "outcome": outcome,
            "steps": steps,
            "duration_s": time.monotonic() - started,
            "task_goal": task_goal,
            "difficulty": difficulty,
            "model_seed": seed,
            "video_path": video_path,
            "server_metadata": client.metadata,
            "memory_stats": memory_stats,
            "error": None,
        }
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        if client is not None:
            client.close()


def worker(args: argparse.Namespace) -> None:
    while True:
        with connect_db(args.db) as db:
            job = claim_job(
                db,
                worker_id=args.worker_id,
                gpu_id=args.gpu_id,
                stale_seconds=args.stale_seconds,
                max_attempts=args.max_attempts,
            )
        if job is None:
            with connect_db(args.db) as db:
                active = db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('pending','running')"
                ).fetchone()[0]
            if active == 0:
                print(f"[{args.worker_id}] all jobs terminal; exiting", flush=True)
                return
            time.sleep(min(30.0, args.stale_seconds / 4))
            continue
        task, episode = str(job["task"]), int(job["episode"])
        print(f"[{args.worker_id}] starting {task} episode {episode} attempt {job['attempts']}", flush=True)
        try:
            result = run_episode(args, job)
            finish_job(args.db, task, episode, args.worker_id, result)
            print(
                f"[{args.worker_id}] finished {task} episode {episode}: {result['status']} "
                f"steps={result['steps']} duration={result['duration_s']:.1f}s",
                flush=True,
            )
        except Exception:
            error = traceback.format_exc()
            print(f"[{args.worker_id}] failed {task} episode {episode}:\n{error}", file=sys.stderr, flush=True)
            retry_or_fail_job(
                args.db,
                task,
                episode,
                worker_id=args.worker_id,
                error=error,
                max_attempts=args.max_attempts,
            )
            time.sleep(args.retry_delay)


def build_summary(db_path: Path) -> dict[str, Any]:
    with connect_db(db_path) as db:
        metadata = meta_dict(db)
        rows = db.execute(
            """
            SELECT task, task_order, status, COUNT(*) count, AVG(duration_s) mean_duration
            FROM jobs GROUP BY task, task_order, status ORDER BY task_order, status
            """
        ).fetchall()
        finished = db.execute(
            "SELECT finished_at FROM jobs WHERE status IN ('success','failure','error') AND finished_at IS NOT NULL"
        ).fetchall()
    task_data: dict[str, dict[str, Any]] = {
        task: {"task": task, "total": 50, "pending": 0, "running": 0, "success": 0, "failure": 0, "error": 0}
        for task in TASKS
    }
    for row in rows:
        task_data[row["task"]][row["status"]] = int(row["count"])
    for values in task_data.values():
        completed = values["success"] + values["failure"]
        values["terminal_accuracy"] = values["success"] / completed if completed else None
    totals = {key: sum(values[key] for values in task_data.values()) for key in ("total", "pending", "running", *TERMINAL)}
    completed = totals["success"] + totals["failure"]
    totals["terminal_accuracy"] = totals["success"] / completed if completed else None
    now = time.time()
    created = float(metadata.get("created_at", now))
    terminal_times = [float(row["finished_at"]) for row in finished]
    totals["rate_all_per_min"] = len(terminal_times) / max(now - created, 1.0) * 60.0
    recent = sum(value >= now - 600 for value in terminal_times)
    totals["rate_10m_per_min"] = recent / 10.0
    remaining = totals["pending"] + totals["running"]
    rate = totals["rate_10m_per_min"] or totals["rate_all_per_min"]
    totals["eta_seconds"] = remaining / rate * 60.0 if rate > 0 else None
    suites: list[dict[str, Any]] = []
    for suite_name, suite_tasks in SUITES.items():
        values = {
            key: sum(task_data[task][key] for task in suite_tasks)
            for key in ("total", "pending", "running", *TERMINAL)
        }
        completed = values["success"] + values["failure"]
        values["terminal_accuracy"] = values["success"] / completed if completed else None
        values["suite"] = suite_name
        suites.append(values)
    return {
        "schema_version": 1,
        "updated_at": utc_now(),
        "meta": metadata,
        "tasks": list(task_data.values()),
        "suites": suites,
        "totals": totals,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_summary(summary: dict[str, Any]) -> None:
    totals = summary["totals"]
    print(f"\nRoboMME summary {summary['updated_at']}")
    print("Task                  Done   S   F   E   R   P    Acc")
    print("--------------------  ---- --- --- --- --- ---  ------")
    for row in summary["tasks"]:
        done = row["success"] + row["failure"] + row["error"]
        accuracy = "   n/a" if row["terminal_accuracy"] is None else f"{100 * row['terminal_accuracy']:5.1f}%"
        print(
            f"{row['task']:<20}  {done:>4} {row['success']:>3} {row['failure']:>3} "
            f"{row['error']:>3} {row['running']:>3} {row['pending']:>3}  {accuracy}"
        )
    terminal = totals["success"] + totals["failure"] + totals["error"]
    accuracy = "n/a" if totals["terminal_accuracy"] is None else f"{100 * totals['terminal_accuracy']:.2f}%"
    print(
        f"TOTAL {terminal}/{totals['total']}  success={totals['success']} failure={totals['failure']} "
        f"error={totals['error']} running={totals['running']} pending={totals['pending']} acc={accuracy}"
    )
    suite_parts = []
    for row in summary["suites"]:
        done = row["success"] + row["failure"]
        accuracy = "n/a" if not done else f"{100 * row['terminal_accuracy']:.1f}%"
        suite_parts.append(f"{row['suite']}={row['success']}/{done} ({accuracy})")
    print("suites " + " | ".join(suite_parts))
    print(
        f"rate all={totals['rate_all_per_min']:.2f}/min last10m={totals['rate_10m_per_min']:.2f}/min "
        f"ETA={format_eta(totals['eta_seconds'])}",
        flush=True,
    )


def wandb_metrics(summary: dict[str, Any]) -> dict[str, float | int]:
    totals = summary["totals"]
    metrics: dict[str, float | int] = {
        "eval/terminal": totals["success"] + totals["failure"] + totals["error"],
        "eval/success": totals["success"],
        "eval/failure": totals["failure"],
        "eval/error": totals["error"],
        "eval/running": totals["running"],
        "eval/pending": totals["pending"],
        "eval/rate_all_per_min": totals["rate_all_per_min"],
        "eval/rate_10m_per_min": totals["rate_10m_per_min"],
    }
    if totals["terminal_accuracy"] is not None:
        metrics["eval/terminal_accuracy"] = totals["terminal_accuracy"]
    if totals["eta_seconds"] is not None:
        metrics["eval/eta_seconds"] = totals["eta_seconds"]
    for row in summary["tasks"]:
        prefix = f"task/{row['task']}"
        metrics[f"{prefix}/success"] = row["success"]
        metrics[f"{prefix}/failure"] = row["failure"]
        metrics[f"{prefix}/error"] = row["error"]
        if row["terminal_accuracy"] is not None:
            metrics[f"{prefix}/terminal_accuracy"] = row["terminal_accuracy"]
    for row in summary["suites"]:
        prefix = f"suite/{row['suite']}"
        metrics[f"{prefix}/success"] = row["success"]
        metrics[f"{prefix}/failure"] = row["failure"]
        metrics[f"{prefix}/error"] = row["error"]
        if row["terminal_accuracy"] is not None:
            metrics[f"{prefix}/terminal_accuracy"] = row["terminal_accuracy"]
    return metrics


def monitor(args: argparse.Namespace) -> None:
    wandb_run = None
    last_terminal = -1
    if args.wandb:
        import wandb

        with connect_db(args.db) as db:
            metadata = meta_dict(db)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=metadata.get("run_name", args.run_name),
            id=metadata["wandb_run_id"],
            resume="allow",
            dir=str(args.results_dir / "wandb"),
            config={"benchmark": "RoboMME", "episodes": 800, "scheduler": "dynamic_episode"},
        )
    while True:
        summary = build_summary(args.db)
        atomic_write_json(args.results_dir / "summary.json", summary)
        print_summary(summary)
        totals = summary["totals"]
        terminal = totals["success"] + totals["failure"] + totals["error"]
        if wandb_run is not None and terminal != last_terminal:
            # Let W&B keep its own monotonic step.  The migrated evaluation DB
            # can legitimately move terminal counts backwards when stale jobs
            # are reset for retry; eval/terminal remains the canonical x-axis.
            wandb_run.log(wandb_metrics(summary))
            last_terminal = terminal
        if args.once or terminal == totals["total"]:
            if wandb_run is not None:
                wandb_run.finish()
            return
        time.sleep(args.interval)


def preflight(args: argparse.Namespace) -> None:
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    import mani_skill
    import openpi_client
    import sapien
    import torch

    del mani_skill, openpi_client, sapien, torch
    counts = {task: BenchmarkEnvBuilder(task, dataset="test").get_episode_num() for task in TASKS}
    if set(counts.values()) != {50} or sum(counts.values()) != 800:
        raise RuntimeError(f"Invalid test metadata counts: {counts}")
    print("metadata: 16 tasks x 50 episodes = 800")
    if args.gpu_smoke:
        from robomme.env_record_wrapper import BenchmarkEnvBuilder
        import robomme.robomme_env  # noqa: F401

        builder = BenchmarkEnvBuilder(TASKS[0], dataset="test", action_space="joint_angle", max_steps=2)
        env = builder.make_env_for_episode(0)
        try:
            obs, info = env.reset()
            print(
                "gpu_smoke:",
                np.asarray(obs["front_rgb_list"][-1]).shape,
                np.asarray(obs["wrist_rgb_list"][-1]).shape,
                len(obs["front_rgb_list"]),
                info["task_goal"][0],
            )
        finally:
            env.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--db", type=Path, required=True)
    init.add_argument("--run-name", required=True)
    init.add_argument("--max-attempts", type=int, default=3)

    work = sub.add_parser("worker")
    work.add_argument("--db", type=Path, required=True)
    work.add_argument("--results-dir", type=Path, required=True)
    work.add_argument("--worker-id", required=True)
    work.add_argument("--gpu-id", type=int, required=True)
    work.add_argument("--server-host", default="127.0.0.1")
    work.add_argument("--server-port", type=int, required=True)
    work.add_argument("--max-steps", type=int, default=1300)
    work.add_argument("--replan-steps", type=int, default=10)
    work.add_argument("--model-seed", type=int, default=42)
    work.add_argument("--max-attempts", type=int, default=3)
    work.add_argument("--stale-seconds", type=int, default=1800)
    work.add_argument("--heartbeat-seconds", type=int, default=30)
    work.add_argument("--retry-delay", type=float, default=5.0)
    work.add_argument("--save-video", choices=("none", "failures", "all"), default="failures")
    work.add_argument("--video-fps", type=int, default=30)
    work.add_argument("--video-stride", type=int, default=2)

    mon = sub.add_parser("monitor")
    mon.add_argument("--db", type=Path, required=True)
    mon.add_argument("--results-dir", type=Path, required=True)
    mon.add_argument("--run-name", default="robomme-eval")
    mon.add_argument("--interval", type=float, default=15.0)
    mon.add_argument("--once", action="store_true")
    mon.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    mon.add_argument("--wandb-project", default="openpi")

    pre = sub.add_parser("preflight")
    pre.add_argument("--gpu-smoke", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "init":
        init_db(args.db, max_attempts=args.max_attempts, run_name=args.run_name)
    elif args.command == "worker":
        if args.replan_steps <= 0 or args.replan_steps > 50:
            raise ValueError("replan_steps must be in [1, 50]")
        if args.video_stride <= 0:
            raise ValueError("video_stride must be positive")
        worker(args)
    elif args.command == "monitor":
        monitor(args)
    elif args.command == "preflight":
        preflight(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
