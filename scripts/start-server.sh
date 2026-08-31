#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="$project_dir/logs"
log_file="$log_dir/qwen-a2a.log"
pid_file="$log_dir/qwen-a2a.pid"
server_bin="${QWEN_A2A_BIN:-$project_dir/.venv/bin/qwen-a2a}"

mkdir -p "$log_dir"

if [[ -f "$pid_file" ]]; then
  existing_pid="$(<"$pid_file")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    printf 'qwen-a2a is already running (PID %s)\n' "$existing_pid"
    printf 'Log: %s\n' "$log_file"
    exit 0
  fi
  rm -f "$pid_file"
fi

if [[ ! -x "$server_bin" ]]; then
  printf 'Executable not found: %s\n' "$server_bin" >&2
  printf "Run: python -m venv .venv && .venv/bin/pip install -e '.[dev]'\n" >&2
  exit 1
fi

cd "$project_dir"
nohup "$server_bin" >>"$log_file" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$pid_file"

sleep 1
if ! kill -0 "$server_pid" 2>/dev/null; then
  rm -f "$pid_file"
  printf 'qwen-a2a failed to start. Recent log output:\n' >&2
  tail -n 20 "$log_file" >&2
  exit 1
fi

printf 'qwen-a2a started (PID %s)\n' "$server_pid"
printf 'Log: %s\n' "$log_file"
printf 'Stop: %s/scripts/stop-server.sh\n' "$project_dir"
