#!/bin/sh
# Остановить серверы, поднятые start.sh (по pid-файлам).
# Не трогает чужие llama-server (например, Bonsai-demo на 8080).
set -e
RUN_DIR="$(cd "$(dirname "$0")" && pwd)/run"

for name in fast think decider; do
    pid_file="$RUN_DIR/$name.pid"
    [ -f "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        # Ждём настоящего выхода. llama-server отдаёт VRAM секунды, и старт, запущенный
        # сразу после kill, либо упрётся в занятую память, либо увидит живой /health
        # у умирающего сервера и решит, что поднимать нечего.
        waited=0
        while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 15 ]; do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "$name не завершился за 15 с — снимаю силой"
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
        echo "остановлен $name (pid $pid)"
    else
        echo "$name уже не работает"
    fi
    rm -f "$pid_file"
done
