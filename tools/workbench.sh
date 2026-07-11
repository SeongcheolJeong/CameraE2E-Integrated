#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$ROOT/.camerae2e"
VENV="$ROOT/.venv"
BACKEND_PID="$STATE_DIR/backend.pid"
FRONTEND_PID="$STATE_DIR/frontend.pid"
BACKEND_LOG="$STATE_DIR/backend.log"
FRONTEND_LOG="$STATE_DIR/frontend.log"
BACKEND_PORT="${CAMERAE2E_BACKEND_PORT:-8010}"
FRONTEND_PORT="${CAMERAE2E_FRONTEND_PORT:-5175}"

usage() {
  cat <<'EOF'
Usage: ./tools/workbench.sh COMMAND

Commands:
  bootstrap  Create the environment, install dependencies, validate, and start
  setup      Create the environment and install Python/Node dependencies
  check      Validate tools, imports, integrated assets, and frontend dependencies
  start      Start backend and frontend in the background
  stop       Stop launcher-managed backend and frontend processes
  restart    Stop and start both processes
  status     Show process, URL, asset, dataset, and model readiness
  logs       Follow backend and frontend logs
EOF
}

version_major() {
  "$1" --version 2>&1 | sed -E 's/[^0-9]*([0-9]+).*/\1/' | head -1
}

require_tool() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: required tool '$1' was not found." >&2; exit 1; }
}

python_command() {
  if [[ -n "${CAMERAE2E_PYTHON:-}" ]]; then
    [[ -x "$CAMERAE2E_PYTHON" ]] || { echo "ERROR: CAMERAE2E_PYTHON is not executable." >&2; exit 1; }
    printf '%s\n' "$CAMERAE2E_PYTHON"
    return
  fi
  if [[ -x "$VENV/bin/python" ]] && \
     "$VENV/bin/python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
    printf '%s\n' "$VENV/bin/python"
    return
  fi
  for candidate in python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "ERROR: Python 3.12 was not found." >&2
  exit 1
}

assert_versions() {
  local python node_major
  python="$(python_command)"
  [[ "$(version_major "$python")" == "3" ]] || { echo "ERROR: Python 3 is required." >&2; exit 1; }
  "$python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || {
    echo "ERROR: CameraE2E requires Python 3.12." >&2
    exit 1
  }
  require_tool node
  require_tool npm
  node_major="$(version_major node)"
  (( node_major >= 20 )) || { echo "ERROR: Node.js 20 or newer is required." >&2; exit 1; }
}

pid_running() {
  [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null
}

cleanup_stale_pid() {
  if [[ -f "$1" ]] && ! pid_running "$1"; then rm -f "$1"; fi
}

port_available() {
  local port="$1"
  ! (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
}

wait_for_url() {
  local url="$1" name="$2"
  for _ in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "ERROR: $name did not become ready at $url." >&2
  return 1
}

setup() {
  assert_versions
  local python
  python="$(python_command)"
  if [[ -x "$VENV/bin/python" ]] && \
     ! "$VENV/bin/python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
    echo "Replacing incompatible $VENV (CameraE2E requires Python 3.12)."
    rm -rf "$VENV"
  fi
  if [[ ! -x "$VENV/bin/python" ]]; then "$python" -m venv "$VENV"; fi
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -e "$ROOT[workbench,yolo,dev]"
  (cd "$ROOT/camerae2e-workbench" && npm ci)
}

check() {
  assert_versions
  [[ -x "$VENV/bin/python" ]] || { echo "ERROR: run './tools/workbench.sh setup' first." >&2; exit 1; }
  "$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
  [[ -d "$ROOT/camerae2e-workbench/node_modules" ]] || { echo "ERROR: frontend dependencies are missing." >&2; exit 1; }
  "$VENV/bin/python" -c 'import camerae2e_v2, fastapi, pyisetcam, uvicorn'
  "$VENV/bin/python" "$ROOT/tools/check_integrated_repository.py"
  echo "CameraE2E preflight passed."
}

start() {
  check
  mkdir -p "$STATE_DIR"
  cleanup_stale_pid "$BACKEND_PID"
  cleanup_stale_pid "$FRONTEND_PID"
  if ! pid_running "$BACKEND_PID" && ! port_available "$BACKEND_PORT"; then
    echo "ERROR: backend port $BACKEND_PORT is already in use by an unmanaged process." >&2
    exit 1
  fi
  if ! pid_running "$FRONTEND_PID" && ! port_available "$FRONTEND_PORT"; then
    echo "ERROR: frontend port $FRONTEND_PORT is already in use by an unmanaged process." >&2
    exit 1
  fi
  if ! pid_running "$BACKEND_PID"; then
    (
      cd "$ROOT/camerae2e-workbench"
      nohup "$VENV/bin/python" -m uvicorn backend.app.main:app \
        --host 127.0.0.1 --port "$BACKEND_PORT" >"$BACKEND_LOG" 2>&1 &
      echo $! >"$BACKEND_PID"
    )
  fi
  if ! pid_running "$FRONTEND_PID"; then
    (
      cd "$ROOT/camerae2e-workbench"
      nohup npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" >"$FRONTEND_LOG" 2>&1 &
      echo $! >"$FRONTEND_PID"
    )
  fi
  if ! wait_for_url "http://127.0.0.1:$BACKEND_PORT/docs" backend || \
     ! wait_for_url "http://127.0.0.1:$FRONTEND_PORT" frontend; then
    echo "Inspect $BACKEND_LOG and $FRONTEND_LOG" >&2
    exit 1
  fi
  echo "Workbench: http://127.0.0.1:$FRONTEND_PORT"
  echo "API docs:  http://127.0.0.1:$BACKEND_PORT/docs"
}

stop_one() {
  local pid_file="$1" name="$2"
  if pid_running "$pid_file"; then
    local pid
    pid="$(cat "$pid_file")"
    kill "$pid"
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    kill -9 "$pid" 2>/dev/null || true
    echo "Stopped $name."
  fi
  rm -f "$pid_file"
}

stop() {
  stop_one "$FRONTEND_PID" frontend
  stop_one "$BACKEND_PID" backend
}

status() {
  cleanup_stale_pid "$BACKEND_PID"
  cleanup_stale_pid "$FRONTEND_PID"
  if pid_running "$BACKEND_PID"; then echo "backend: running (PID $(cat "$BACKEND_PID"))"; else echo "backend: stopped"; fi
  if pid_running "$FRONTEND_PID"; then echo "frontend: running (PID $(cat "$FRONTEND_PID"))"; else echo "frontend: stopped"; fi
  echo "KITTI root: ${CAMERAE2E_KITTI_ROOT:-not configured}"
  echo "YOLO model: ${CAMERAE2E_YOLO_MODEL:-not configured}"
  echo "Projects:   ${CAMERAE2E_PROJECTS_ROOT:-$ROOT/camerae2e-workbench/projects}"
}

logs() {
  mkdir -p "$STATE_DIR"
  touch "$BACKEND_LOG" "$FRONTEND_LOG"
  tail -f "$BACKEND_LOG" "$FRONTEND_LOG"
}

case "${1:-}" in
  bootstrap) setup; check; start ;;
  setup) setup ;;
  check) check ;;
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) logs ;;
  *) usage; [[ -n "${1:-}" ]] && exit 2 || exit 0 ;;
esac
