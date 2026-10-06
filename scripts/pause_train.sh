#!/usr/bin/env bash
# Espera a que termine la epoca indicada y pausa (SIGSTOP) el entrenamiento.
#   ./scripts/pause_train.sh 32
# Reanudar:
#   kill -CONT -<PGID>     (el PGID se imprime al pausar)
set -u
cd "$(dirname "$0")/.."

EP=${1:?uso: $0 <epoca_que_debe_terminar>  p.ej. 32}
LOG=${LOG:-logs/train_240.log}
NEXT=$(printf '%03d' "$((EP + 1))")

linea=$(ps -eo pgid=,args= | grep -F 'scripts/train.py' | grep -v grep | head -1)
[ -z "$linea" ] && { echo "ERROR: no hay proceso de entrenamiento en marcha"; exit 1; }
PGID=$(awk '{print $1}' <<<"$linea")

echo "PGID=$PGID"
echo "pausare al empezar la epoca $NEXT (= fin de la $EP) | log=$LOG"

while :; do
    if grep -qE "^  ep${NEXT}[[:space:]]+[0-9]+/" "$LOG" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$PGID" 2>/dev/null; then
        echo "el proceso dejo de existir antes de tiempo"; exit 1
    fi
    sleep 3
done

kill -STOP -- "-$PGID" 2>/dev/null || kill -STOP "$PGID"
echo
echo "PAUSADO tras la epoca $EP"
echo "reanudar con:  kill -CONT -$PGID"
