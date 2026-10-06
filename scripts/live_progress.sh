#!/bin/bash
# Log de progreso en vivo (sin TTY): tail -f logs/live_progress.log
cd "$(dirname "$0")/.."
: > logs/live_progress.log
while pgrep -f "scripts/fetch_inat.py fetch|scripts/fetch_medfish.py|scripts/fetch_biota.py" > /dev/null 2>&1; do
  echo "$(date +%H:%M:%S) ================" >> logs/live_progress.log
  .venv-train/bin/python scripts/progress.py >> logs/live_progress.log 2>&1
  echo >> logs/live_progress.log
  sleep 15
done
echo "$(date +%H:%M:%S) ================ TODAS LAS DESCARGAS TERMINADAS" >> logs/live_progress.log
.venv-train/bin/python scripts/progress.py >> logs/live_progress.log 2>&1
