#!/usr/bin/env python3
"""Cruza las especies de PejeDex (seedData.ts) con el dataset y sondea iNat
para ver cuales se podrian añadir (taxon resoluble + fotos con licencia).

Salida: data_raw/inat_app_probe.json
"""
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fetch_inat as F  # noqa: E402

SEED = "/home/aithor/Escritorio/PejeDex/PejeDex/src/db/seedData.ts"
OUT = os.path.join(ROOT, "data_raw", "inat_app_probe.json")
NC = "cc-by,cc-by-sa,cc0,pdm,cc-by-nc,cc-by-nc-sa"
# familias que no interesan a un pescador de costa: fondo/hondura y oceánicas
SKIP = {"Myctophidae","Stomiidae","Centrophoridae","Melamphaidae","Macrouridae",
        "Nettastomatidae","Synaphobranchidae","Ophidiidae","Alepocephalidae",
        "Sternoptychidae","Trachichthyidae","Somniosidae","Dalatiidae","Etmopteridae",
        "Lampridae","Regalecidae","Alepisauridae","Trichiuridae","Bramidae","Gempylidae",
        "Lophiidae","Carapidae","Moridae","Chlopsidae","Istiophoridae","Exocoetidae",
        "Coryphaenidae","Cetorhinidae","Rhincodontidae","Alopiidae","Lamnidae",
        "Pseudocarchariidae","Odontaspididae","Berycidae","Centrolophidae",
        "Caproidae","Chaetodontidae","Polyprionidae","Holocentridae"}


def load_app_species():
    txt = open(SEED, encoding="utf-8").read()
    return re.findall(
        r'\{\s*commonName:\s*"([^"]*)"\s*,\s*scientificName:\s*"([^"]*)"\s*,'
        r'\s*family:\s*"([^"]*)"\s*,\s*minSize:\s*([0-9.]+)',
        txt,
    )


def split2(x):
    p = x.split()
    return (p[0].lower(), p[1].lower()) if len(p) >= 2 else (x.lower(), "")


def close(a, b):
    if a[0] != b[0]:
        return False
    if a[1] == b[1]:
        return True
    return SequenceMatcher(None, a[1], b[1]).ratio() > 0.80


def main():
    app = load_app_species()
    classes = set(json.load(open(os.path.join(ROOT, "dataset", "classes.json"),
                                 encoding="utf-8"))["classes"])
    print(f"app={len(app)}  dataset={len(classes)}", flush=True)

    exact, mapped, missing = [], [], []
    for common, sci, fam, mins in app:
        rec = {"common": common, "sci": sci, "family": fam,
               "min_size": float(mins)}
        if sci in classes:
            exact.append(rec)
            continue
        hit = next((c for c in classes if close(split2(sci), split2(c))), None)
        if hit:
            rec["dataset_as"] = hit
            mapped.append(rec)
        else:
            missing.append(rec)

    print(f"exactas={len(exact)} sinonimia={len(mapped)} faltan={len(missing)}",
          flush=True)
    n_all = len(missing)
    missing = [r for r in missing if r["family"] not in SKIP]
    print(f"de esas, saltando {len(SKIP)} familias de fondo/oceano -> "
          f"sondear {len(missing)} (antes {n_all})", flush=True)
    for r in mapped:
        print(f"  SINONIMIA: {r['sci']:36s} == {r['dataset_as']}", flush=True)

    # reutiliza cache previa
    cache = {}
    if os.path.exists(OUT):
        cache = json.load(open(OUT, encoding="utf-8")).get("cache", {})

    t0 = time.time()
    for i, r in enumerate(missing, 1):
        sci = r["sci"]
        if sci in cache:
            r.update(cache[sci])
            continue
        tid = com = None
        err = None
        for attempt in range(4):
            try:
                tid, resolved, com = F.resolve_taxon(sci)
                break
            except Exception as e:  # noqa: BLE001
                err = str(e)
                time.sleep(2 * (attempt + 1))
        info = {"taxon_id": tid, "resolved": resolved if tid else None,
                "common_en": com, "error": err if not tid else None}
        F.OPEN_LICENSES = NC
        try:
            info["cc_res_nc"] = F.count_photos(tid, True) if tid else 0
        except Exception as e:  # noqa: BLE001
            info["cc_res_nc"] = 0
            info["err1"] = str(e)
        finally:
            F.OPEN_LICENSES = "cc-by,cc-by-sa,cc0,pdm"
        info["cc_res"] = info["cc_res_nc"]
        r.update(info)
        cache[sci] = info
        if i % 5 == 0 or i == len(missing):
            print(f"  [{i}/{len(missing)}] {time.time()-t0:.0f}s  "
                  f"{sci} -> id={tid} fotos_nc={info.get('cc_res_nc')}",
                  flush=True)
            json.dump({"app_total": len(app), "exactas": exact,
                       "sinonimia": mapped, "faltan": missing, "cache": cache},
                      open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    json.dump({"app_total": len(app), "exactas": exact, "sinonimia": mapped,
               "faltan": missing, "cache": cache},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    ok = [r for r in missing if r.get("taxon_id")]
    enoug = [r for r in ok if (r.get("cc_res_nc") or 0) >= 90]
    print(f"\nRESULTADO: faltan={len(missing)}  resueltas={len(ok)}  "
          f"con >=90 fotos={len(enoug)}", flush=True)
    print("\n-- añadibles con >=90 fotos (nc) --", flush=True)
    for r in sorted(enoug, key=lambda x: -(x["cc_res_nc"] or 0)):
        print(f"  {r['common'][:24]:24s} {r['sci']:34s} "
              f"fotos={r['cc_res_nc']:4d} minSize={r['min_size']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
