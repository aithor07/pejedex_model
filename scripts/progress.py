#!/usr/bin/env python3
"""Progreso de descarga de datos de PejeDex.

Uso:
  python scripts/progress.py            # una vez
  watch -n 10 python scripts/progress.py
  python scripts/progress.py --top 15   # tambien desglose por especie
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data_raw")
LOGS = os.path.join(ROOT, "logs")

MEDFISH_TARGET = 69153


def count_images(folder):
    if not os.path.isdir(folder):
        return 0, 0
    n = b = 0
    for dp, _dn, fn in os.walk(folder):
        for f in fn:
            if f.endswith((".jpg", ".jpeg", ".png")):
                n += 1
                try:
                    b += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
    return n, b


def tail(path, lines=40):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 20000))
        return f.read().decode("utf-8", "replace").splitlines()[-lines:]


def medfish_rate():
    """Lee 'img/s' de la ultima linea util del log, si existe."""
    for line in reversed(tail(os.path.join(LOGS, "medfish.log"))):
        m = re.search(r"(\d+)/(\d+) ok=(\d+) fail=(\d+) ([\d.]+) img/s", line)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))
    return None


def fill_progress(inat_n):
    """Barra del relleno de licencias CC-BY-NC (si hay estado guardado)."""
    st_path = os.path.join(LOGS, "inat_fill_progress.json")
    if not os.path.exists(st_path):
        return None
    try:
        st = json.load(open(st_path, encoding="utf-8"))
    except Exception:
        return None
    base, target, start = st["base"], st["target"], st["start"]
    want = target - base
    got = max(0, inat_n - base)
    if want <= 0:
        return None
    elapsed = max(1e-6, time.time() - start)
    ips = got / elapsed
    bar_w = 32
    done = int(bar_w * min(got, want) / want)
    eta = (want - got) / ips if ips and got < want else 0
    line = f"  RELLENO   [{('#' * done).ljust(bar_w)}] {100.0*got/want:5.1f}%  "
    line += f"nuevas {got:,}/{want:,} · {ips:4.1f} img/s"
    line += f" · ETA {fmt_eta(eta) if got < want else 'hecho'}"
    line += f" · lleva {fmt_eta(elapsed)}"
    log = os.path.join(LOGS, "inat_fill.log")
    if os.path.exists(log):
        done_sp = sum(1 for l in tail(log, 400) if "nuevas=" in l)
        if done_sp:
            line += f" · especies {done_sp}/{st.get('target_species', '?')}"
    try:
        alive = subprocess.run(["pgrep", "-f", "scripts/fetch_inat.py fetch"],
                               capture_output=True, text=True).stdout.split()
        line += " · ACTIVA" if alive else " · (proceso parado)"
    except Exception:
        pass
    return line


def fmt_eta(sec):
    if sec is None or sec != sec or sec < 0:
        return "--"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="desglose por especie")
    ap.add_argument("--pid-check", action="store_true", help="indica si la descarga sigue activa")
    args = ap.parse_args()

    med_n, med_b = count_images(os.path.join(RAW, "medfish"))
    inat_n, inat_b = count_images(os.path.join(RAW, "inat"))
    biota_n, biota_b = count_images(os.path.join(RAW, "biota"))

    total_n = med_n + inat_n + biota_n
    total_b = med_b + inat_b + biota_b

    rate = medfish_rate()
    now = time.strftime("%H:%M:%S")
    print(f"PejeDex · datos · {now}")
    print("-" * 62)

    pct = 100.0 * med_n / MEDFISH_TARGET
    bar_w = 32
    done = int(bar_w * med_n / MEDFISH_TARGET)
    print(f"MEDFISH101  [{('#' * done).ljust(bar_w)}] {pct:5.1f}%  "
          f"{med_n:,}/{MEDFISH_TARGET:,}  {med_b/1024**3:5.1f} GB")
    if rate:
        done_r, total_r, ok, fail, ips = rate
        try:
            med_alive = bool(subprocess.run(["pgrep", "-f", "scripts/fetch_medfish.py"],
                                            capture_output=True, text=True).stdout.split())
        except Exception:
            med_alive = True
        if med_alive:
            remain = total_r - done_r
            eta = remain / ips if ips else None
            print(f"            {ips:4.1f} img/s · fallos {fail} ({100*fail/max(1,done_r):.1f}%) · "
                  f"ETA {fmt_eta(eta)}")
        else:
            print(f"            TERMINADO · {ok:,} ok · {fail} enlaces muertos "
                  f"({100*fail/max(1,done_r):.1f}%) · 18.6 img/s")
    else:
        print("            sin linea de ritmo en el log (¿terminado o no arrancado?)")

    print(f"iNaturalist  {inat_n:>7,} img  {inat_b/1024**3:5.1f} GB")
    fill = fill_progress(inat_n)
    if fill:
        print(fill)
    print(f"BIOTA        {biota_n:>7,} img  {biota_b/1024**3:5.2f} GB")
    print("-" * 62)
    print(f"TOTAL        {total_n:>7,} img  {total_b/1024**3:5.1f} GB")

    free = os.statvfs(ROOT)
    free_gb = free.f_bavail * free.f_frsize / 1024**3
    print(f"disco libre  {free_gb:5.1f} GB")

    if args.pid_check:
        try:
            out = subprocess.run(["pgrep", "-f", "scripts/fetch_medfish.py|scripts/fetch_inat.py|scripts/fetch_biota.py"],
                                 capture_output=True, text=True).stdout.split()
            print(f"procesos de descarga activos: {len(out)}")
        except Exception:
            pass

    if args.top:
        per = Counter()
        man = os.path.join(RAW, "medfish_manifest.jsonl")
        if os.path.exists(man):
            with open(man, encoding="utf-8") as f:
                for line in f:
                    try:
                        per[json.loads(line)["species"]] += 1
                    except Exception:
                        pass
        if per:
            print("-" * 62)
            print(f"top {args.top} especies ya descargadas:")
            for sp, n in per.most_common(args.top):
                print(f"  {n:>5,}  {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
