from __future__ import annotations

import socket

import psutil
import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def _pid_on_port(port: int) -> int | None:
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr.port == port and conn.status == "LISTEN":
            return conn.pid
    return None


def _kill_existing_server() -> None:
    pid = _pid_on_port(PORT)
    if pid is None:
        return
    try:
        proc = psutil.Process(pid)
        # Only kill a process that is actually a running server on this port.
        name = proc.name()
        print(f"Killing existing process on {HOST}:{PORT} (pid={pid}, name={name})")
        proc.terminate()
        proc.wait(timeout=5)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as exc:
        print(f"Could not stop existing process pid={pid}: {exc}")


if __name__ == "__main__":
    _kill_existing_server()
    # Re-check the port is now free; start the server regardless.
    # This is the local development entry point. Reloading ensures generation
    # requests never keep using stale telemetry code after a source edit.
    uvicorn.run("app.api.main:app", host=HOST, port=PORT, reload=True)
