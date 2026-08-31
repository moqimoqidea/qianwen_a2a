#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pid_file="$project_dir/logs/qwen-a2a.pid"

if [[ ! -f "$pid_file" ]]; then
  printf 'qwen-a2a is not running: PID file not found.\n'
  exit 0
fi

server_pid="$(<"$pid_file")"
if [[ ! "$server_pid" =~ ^[0-9]+$ ]]; then
  printf 'Invalid PID file: %s\n' "$pid_file" >&2
  exit 1
fi

if ! kill -0 "$server_pid" 2>/dev/null; then
  rm -f "$pid_file"
  printf 'Removed stale PID file; qwen-a2a was not running.\n'
  exit 0
fi

command_line="$(ps -p "$server_pid" -o command= 2>/dev/null || true)"
if [[ "$command_line" != *qwen-a2a* && "$command_line" != *qwen_a2a* ]]; then
  printf 'PID %s does not appear to be qwen-a2a; refusing to stop it.\n' "$server_pid" >&2
  exit 1
fi

kill "$server_pid"
for _ in {1..20}; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    rm -f "$pid_file"
    printf 'qwen-a2a stopped (PID %s)\n' "$server_pid"
    exit 0
  fi
  sleep 0.25
done

printf 'qwen-a2a did not stop within 5 seconds (PID %s).\n' "$server_pid" >&2
exit 1
