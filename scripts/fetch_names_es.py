#!/usr/bin/env python3
"""Obtiene el nombre comun en espanol de cada especie desde iNaturalist.

Salida: data_raw/names_es.json  { "Diplodus sargus": "Sargo", ... }
Se usa para rellenar dataset/classes.json (nombres visibles en la app).
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.inaturalist.org/v1"
UA = "PejeDex-dataset/1.0"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data_raw")
OUT = os.path.join(RAW, "names_es.json")


def get(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "es"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(last)


def main():
    classes = sys.argv[1:]
    if not classes:
        cp = os.path.join(ROOT, "dataset", "classes.json")
        if not os.path.exists(cp):
            sys.exit("dame especies como argumento o ejecuta build_dataset.py antes")
        classes = json.load(open(cp, encoding="utf-8"))["classes"]

    out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for i, sp in enumerate(classes, 1):
        if sp in out and out[sp]:
            continue
        try:
            d = get(f"{API}/taxa?" + urllib.parse.urlencode(
                {"q": sp, "rank": "species", "per_page": 8, "locale": "es"}))
            name = ""
            for t in d.get("results", []):
                if t.get("name", "").lower() == sp.lower():
                    name = t.get("preferred_common_name") or ""
                    break
            if not name and d.get("results"):
                name = d["results"][0].get("preferred_common_name") or ""
            out[sp] = name
            print(f"[{i}/{len(classes)}] {sp:35s} {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] {sp} ERR {e}", flush=True)
            out[sp] = out.get(sp, "")
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        time.sleep(0.3)

    print(f"\n{sum(1 for v in out.values() if v)}/{len(out)} con nombre en espanol -> {OUT}")

    # refrescar dataset/classes.json si ya existe
    cp = os.path.join(ROOT, "dataset", "classes.json")
    if os.path.exists(cp):
        try:
            cj = json.load(open(cp, encoding="utf-8"))
            cj["common_es"] = {c: out.get(c, cj.get("common_es", {}).get(c, ""))
                               for c in cj.get("classes", [])}
            json.dump(cj, open(cp, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            n = sum(1 for v in cj["common_es"].values() if v)
            print(f"actualizado {cp} ({n}/{len(cj['common_es'])} con nombre)")
        except Exception as e:  # noqa: BLE001
            print(f"no se pudo actualizar classes.json: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
