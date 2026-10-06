#!/usr/bin/env python3
"""Descarga las especies costeras nuevas elegidas a partir de inat_app_probe.json.

Reutiliza fetch_inat.cmd_fetch pero con:
  - candidatas = seleccion (no species_candidates.json)
  - probe      = inat_app_probe.json (castellano -> datos ya sondeados)
  - licencias  = permisivas + NC (las que ya usamos en la 2a pasada)

Uso:  python scripts/fetch_add.py [--max 400] [--jobs 12]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import fetch_inat as F  # noqa: E402

APP_PROBE = os.path.join(ROOT, "data_raw", "inat_app_probe.json")
OUT_PROBE = os.path.join(ROOT, "data_raw", "inat_add_probe.json")
SELECTION = os.path.join(HERE, "species_add.json")
NC = "cc-by,cc-by-sa,cc0,pdm,cc-by-nc,cc-by-nc-sa"


def load_selection():
    """Especie -> nombre comun ES de la app."""
    return [c["key"] for c in json.load(open(SELECTION, encoding="utf-8"))["candidates"]]


def build_probe(sel):
    """Dict esperado por cmd_fetch: key -> {taxon_id, cc_research, cc_any}."""
    app = json.load(open(APP_PROBE, encoding="utf-8"))
    by_sci = {r["sci"]: r for r in app["faltan"]}
    out = {}
    for sci in sel:
        r = by_sci.get(sci)
        if not r or not r.get("taxon_id"):
            print(f"  !! {sci}: sin taxon, no se podra descargar")
            continue
        n = r.get("cc_res_nc") or 0
        out[sci] = {"taxon_id": r["taxon_id"], "resolved": r.get("resolved") or sci,
                    "common": r.get("common_en"), "cc_research": n, "cc_any": n}
    json.dump(out, open(OUT_PROBE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--min-count", type=int, default=90)
    ap.add_argument("--skip-csv", default="")
    args = ap.parse_args()

    sel = load_selection()
    print(f"seleccion: {len(sel)} especies")
    probe = build_probe(sel)
    print(f"con taxon: {len(probe)}/{len(sel)}")
    faltan = [s for s in sel if s not in probe]
    if faltan:
        print("sin taxon:", ", ".join(faltan))

    # monta el entorno que espera cmd_fetch
    F.candidates = lambda: [{"key": k} for k in sorted(probe)]
    F.PROBE = OUT_PROBE
    F.OPEN_LICENSES = NC
    # cmd_fetch relee F.PROBE por nombre de modulo
    return F.cmd_fetch(args)


if __name__ == "__main__":
    sys.exit(main())
