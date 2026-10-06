#!/usr/bin/env python3
"""Progreso de la descarga de especies nuevas (scripts/fetch_add.py).

  python scripts/fetch_progress.py           # seguimiento en vivo
  python scripts/fetch_progress.py --once    # una sola foto
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "logs", "fetch_add.log")
MANIFEST = os.path.join(ROOT, "data_raw", "inat_manifest.jsonl")
SEL = os.path.join(HERE, "species_add.json")
LINE = re.compile(r"^(.+?)\s+nuevas=(\d+)\s+fail=(\d+)\s+total=(\d+)\s*$")


def load():
    cands = json.load(open(SEL, encoding="utf-8"))["candidates"]
    sel = [c["key"] for c in cands]
    # objetivo por especie: se decide mirando disco, no el log, para que
    # aguante reinicios del proceso de descarga
    targets = {c["key"]: min(c["photos"], 400) for c in cands}
    objetivo = sum(targets.values())
    done, fails = {}, {}
    if os.path.exists(LOG):
        for ln in open(LOG, encoding="utf-8", errors="replace"):
            m = LINE.match(ln.strip())
            if m:
                done[m.group(1)] = int(m.group(2))
                fails[m.group(1)] = int(m.group(3))
    have = {s: 0 for s in sel}
    # cuenta ficheros en disco: el manifiesto solo se escribe al terminar cada
    # especie, asi que no sirve para estimar el ritmo entre especies.
    base = os.path.join(ROOT, "data_raw", "inat")
    for s in sel:
        d = os.path.join(base, s.replace(" ", "_"))
        if not os.path.isdir(d):
            continue
        try:
            have[s] = sum(1 for e in os.scandir(d)
                          if e.is_file() and e.name.endswith((".jpg", ".jpeg", ".png")))
        except OSError:
            pass
    return sel, objetivo, done, fails, have, targets


def fmt_dur(s):
    if s is None or s != s or s < 0:
        return "?"
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def snapshot():
    sel, objetivo, done, fails, have, targets = load()
    descargadas = sum(have.values())
    completas = [s for s in sel if have[s] >= targets[s]]
    pendientes = [s for s in sel if have[s] < targets[s]]
    # --max 400 puede dejar especies POR ENCIMA del objetivo del probe, asi que
    # el hueco real es la suma de lo que falta en las que aun estan por debajo
    restante = sum(max(0, targets[s] - have[s]) for s in sel)
    hechas = max(0, objetivo - restante)
    return {
        "sel": sel, "objetivo": objetivo, "descargadas": descargadas,
        "hechas": min(hechas, objetivo), "restante": restante,
        "especies_total": len(sel), "especies_ok": len(completas),
        "fails": sum(fails.values()),
        "pendientes": pendientes,
        "have": have, "done": done,
    }


def eta_of(hist, restante):
    """ETA sobre el hueco que queda (no sobre lo ya descargado, que se pasa
    en algunas especies). El manifiesto solo se escribe al terminar cada
    especie, asi que usa una ventana movil."""
    if not hist or restante <= 0:
        return None
    # ventana: usa la muestra mas antigua, salvo que haya una de >=30 s
    t_end, r_end = hist[-1]
    t0, r0 = hist[0]
    for s in hist:
        if t_end - s[0] >= 30:
            t0, r0 = s
            break
    dt, dr = t_end - t0, r0 - r_end   # dr = cuanto ha bajado el hueco
    if dt <= 1 or dr <= 0:
        return None
    return restante * dt / dr


def line(st, t0, hist):
    n, obj, rest = st["descargadas"], st["objetivo"], st["restante"]
    pct = 100 * st["hechas"] / obj if obj else 0
    pct = max(0.0, min(100.0, pct))
    epct = 100 * st["especies_ok"] / st["especies_total"]
    eta = eta_of(hist, rest)
    bar = "#" * int(pct / 2.5)
    out = [f"[{fmt_dur(time.time()-t0)}] descarga iNat",
           f"  hueco      {rest:6d} fotos por bajar (van {n} en disco)",
           f"  cobertura  ({pct:4.1f}%)  |{bar:<40s}|",
           f"  especies   {st['especies_ok']}/{st['especies_total']} "
           f"({epct:4.1f}%)  fallidas/fotos rotas: {st['fails']}",
           f"  ETA        {fmt_dur(eta)}"]
    if st["pendientes"]:
        nxt = ", ".join(st["pendientes"][:4])
        out.append(f"  siguientes {nxt}{' ...' if len(st['pendientes'])>4 else ''}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=15)
    args = ap.parse_args()

    t0 = time.time()
    hist = []
    while True:
        st = snapshot()
        hist.append((time.time(), st["restante"]))
        hist = [h for h in hist if time.time() - h[0] <= 180]
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.write(line(st, t0, hist) + "\n")
        sys.stdout.flush()
        if args.once:
            # dos muestras para poder estimar el ritmo
            t_anchor, n_anchor = time.time(), st["restante"]
            time.sleep(12)
            st = snapshot()
            hist = [(t_anchor, n_anchor), (time.time(), st["restante"])]
            sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(line(st, t0, hist) + "\n")
            sys.stdout.flush()
            return 0
        if st["especies_ok"] >= st["especies_total"] or not st["pendientes"]:
            print("\nDESCARGA COMPLETADA")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
