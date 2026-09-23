#!/bin/sh
# Остановить серверы, поднятые start.sh (по pid-файлам).
# Не трогает чужие llama-server (например, Bonsai-demo на 8080).
set -e
RUN_DIR="$(cd "$(dirname "$0")" && pwd)/run"

for name in fast think; do
    pid_file="$RUN_DIR/$name.pid"
    [ -f "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && echo "остановлен $name (pid $pid)"
    else
        echo "$name уже не работает"
    fi
    rm -f "$pid_file"
done
