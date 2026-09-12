from __future__ import annotations

import os
import time
from pathlib import Path

import psutil
import uvicorn

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
SHUTDOWN_TIMEOUT_SECONDS = 8.0
PORT_RELEASE_TIMEOUT_SECONDS = 5.0
SCRIPT_PATH = Path(__file__).resolve()


def _listener_pids(port: int) -> set[int]:
    pids: set[int] = set()
    for conn in psutil.net_connections(kind="tcp"):
        if (
            conn.pid is not None
            and conn.laddr
            and conn.laddr.port == port
            and conn.status == psutil.CONN_LISTEN
        ):
            pids.add(conn.pid)
    return pids


def _runs_this_script(proc: psutil.Process) -> bool:
    """Return whether *proc* was launched with this project's run.py."""
    try:
        if not proc.name().casefold().startswith("python"):
            return False
        cwd = Path(proc.cwd())
        args = proc.cmdline()[1:]
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False

    if not args or args[0].startswith("-"):
        return False

    expected = os.path.normcase(str(SCRIPT_PATH))
    candidate = Path(args[0])
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    return os.path.normcase(str(resolved)) == expected


def _old_server_roots() -> list[psutil.Process]:
    """Find surviving launchers that explicitly run this project's run.py."""
    current_pid = os.getpid()
    try:
        current_lineage = {
            current_pid,
            *(parent.pid for parent in psutil.Process(current_pid).parents()),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        current_lineage = {current_pid}

    launchers: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid"]):
        if proc.pid not in current_lineage and _runs_this_script(proc):
            launchers.append(proc)
    return launchers


def _stop_process_tree(root: psutil.Process) -> None:
    """Stop one previous reloader and every worker/resource it owns."""
    current_pid = os.getpid()
    try:
        descendants = root.children(recursive=True)
        processes = [
            proc for proc in (root, *descendants) if proc.pid != current_pid
        ]
        name = root.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return

    print(
        f"Stopping previous GenExam process tree "
        f"(root pid={root.pid}, name={name}, processes={len(processes)})"
    )

    # Stop the reloader first so it cannot replace a worker while cleanup runs.
    for proc in processes:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    _, alive = psutil.wait_procs(processes, timeout=SHUTDOWN_TIMEOUT_SECONDS)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=SHUTDOWN_TIMEOUT_SECONDS)


def _wait_for_port_release(port: int) -> set[int]:
    deadline = time.monotonic() + PORT_RELEASE_TIMEOUT_SECONDS
    remaining = _listener_pids(port)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = _listener_pids(port)
    return remaining


def _clean_previous_run() -> None:
    """Release all processes and memory owned by earlier launcher runs."""
    for root in _old_server_roots():
        _stop_process_tree(root)

    remaining = _wait_for_port_release(PORT)
    if remaining:
        owners = ", ".join(str(pid) for pid in sorted(remaining))
        raise RuntimeError(
            f"Cannot start GenExam: {HOST}:{PORT} is still owned by PID(s) "
            f"{owners}. The launcher will not terminate unrelated processes."
        )


if __name__ == "__main__":
    _clean_previous_run()
    uvicorn.run("app.api.main:app", host=HOST, port=PORT)
