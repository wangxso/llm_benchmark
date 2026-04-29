"""Background task manager for long-running operations.

Allows eval, load test, and auto-tune tasks to run in background threads
so the Streamlit UI remains responsive. Progress is tracked in session_state
and displayed on the Results page.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import streamlit as st


@dataclass
class TaskInfo:
    """Tracks a background task."""
    task_id: str
    task_type: str          # "eval" | "loadtest" | "autotune"
    label: str              # Display name
    status: str = "running" # "running" | "completed" | "failed" | "stopped"
    progress: float = 0.0   # 0.0-1.0
    progress_text: str = ""
    result: Optional[Dict] = field(default=None)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = field(default=None, repr=False)

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def elapsed_str(self) -> str:
        s = int(self.elapsed())
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m"


def _get_tasks() -> Dict[str, TaskInfo]:
    if "bg_tasks" not in st.session_state:
        st.session_state["bg_tasks"] = {}
    return st.session_state["bg_tasks"]


def start_task(
    task_type: str,
    label: str,
    target_fn: Callable,
    **kwargs,
) -> str:
    """Start a background task.

    Args:
        task_type: "eval", "loadtest", or "autotune"
        label: Human-readable task name
        target_fn: Function to run in background. Signature:
                   target_fn(task_id: str, stop_event: threading.Event, **kwargs)
        **kwargs: Passed to target_fn

    Returns:
        task_id
    """
    tasks = _get_tasks()

    # Stop any existing running task of the same type
    for tid, t in list(tasks.items()):
        if t.task_type == task_type and t.status == "running":
            t.stop_event.set()
            t.status = "stopped"

    task_id = str(uuid.uuid4())[:8]
    task = TaskInfo(
        task_id=task_id,
        task_type=task_type,
        label=label,
    )

    def _wrapper():
        try:
            target_fn(task_id=task_id, stop_event=task.stop_event, **kwargs)
        except Exception as e:
            _fail(task_id, str(e))

    task.thread = threading.Thread(target=_wrapper, daemon=True)
    tasks[task_id] = task
    task.thread.start()
    return task_id


def get_active_tasks() -> Dict[str, TaskInfo]:
    tasks = _get_tasks()
    return {k: v for k, v in tasks.items() if v.status == "running"}


def get_all_tasks() -> Dict[str, TaskInfo]:
    return _get_tasks()


def get_task(task_id: str) -> Optional[TaskInfo]:
    return _get_tasks().get(task_id)


def stop_task(task_id: str):
    tasks = _get_tasks()
    task = tasks.get(task_id)
    if task and task.status == "running":
        task.stop_event.set()
        task.status = "stopped"
        task.progress_text = "Stopped by user"


def update_progress(task_id: str, progress: float, text: str = ""):
    tasks = _get_tasks()
    task = tasks.get(task_id)
    if task:
        task.progress = min(max(progress, 0.0), 1.0)
        task.progress_text = text


def complete_task(task_id: str, result: Any = None):
    tasks = _get_tasks()
    task = tasks.get(task_id)
    if task:
        task.status = "completed"
        task.progress = 1.0
        task.progress_text = "Completed"
        task.result = result


def _fail(task_id: str, error: str):
    tasks = _get_tasks()
    task = tasks.get(task_id)
    if task:
        task.status = "failed"
        task.progress_text = "Failed"
        task.error = error


def has_active_tasks() -> bool:
    return len(get_active_tasks()) > 0
