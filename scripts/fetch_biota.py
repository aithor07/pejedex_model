#!/usr/bin/env python3
"""Descarga las imagenes BIOTA (Gobierno de Canarias) de las especies objetivo.

Entrada: data_raw/biota_index.json  (17.727 imagenes indexadas)
Salida : data_raw/biota/<Especie>/<image_id>.jpg
         data_raw/biota_manifest.jsonl
"""
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data_raw")
IDX = os.path.join(RAW, "biota_index.json")
DIR = os.path.join(RAW, "biota")
MANIFEST = os.path.join(RAW, "biota_manifest.jsonl")
BASE = "https://www.biodiversidadcanarias.es"

# nombres que usa BIOTA y no coinciden con los nuestros
SYNONYMS = {
    "Coris melanura": "Coris julis",
    "Oblada melanurus": "Oblada melanura",
}


def target_classes():
    classes = set()
    for name, path in (("medfish", os.path.join(RAW, "medfish_links.csv")),):
        if os.path.exists(path):
            import csv
            with open(path, encoding="utf-8") as f:
                classes |= {r["class_name"] for r in csv.DictReader(f)}
    probe = os.path.join(RAW, "inat_probe.json")
    if os.path.exists(probe):
        classes |= set(json.load(open(probe, encoding="utf-8")))
    return classes


def download(url, path, tries=3, timeout=60):
    if os.path.exists(path) and os.path.getsize(path) > 4096:
        return "skip"
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) < 4096:
                return "fail:tiny"
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            return "ok"
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 * (i + 1))
    return f"fail:{last}"


def main():
    os.makedirs(DIR, exist_ok=True)
    targets = target_classes()
    idx = json.load(open(IDX, encoding="utf-8"))

    selected = []
    for it in idx:
        name = SYNONYMS.get(it["scientific"], it["scientific"])
        if name in targets:
            selected.append({**it, "class_name": name})
    print(f"objetivo: {len(targets)} clases · BIOTA: {len(selected)} imagenes seleccionadas")

    by = defaultdict(list)
    for it in selected:
        by[it["class_name"]].append(it)

    manifest = []
    if os.path.exists(MANIFEST):
        manifest = [json.loads(l) for l in open(MANIFEST, encoding="utf-8") if l.strip()]
    have = {(m["species"], m["photo_id"]) for m in manifest}

    jobs = []
    for cls, items in by.items():
        folder = os.path.join(DIR, cls.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)
        for it in items:
            if (cls, it["image_id"]) in have:
                continue
            url = f"{BASE}/biota/especie/{it['species_code']}/imagenes/{it['image_id']}"
            path = os.path.join(folder, f"{it['image_id']}.jpg")
            jobs.append((cls, it["image_id"], url, path))

    print(f"pendientes: {len(jobs)}")
    stats = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(download, u, p): (c, i, p) for c, i, u, p in jobs}
        for fut in as_completed(futs):
            cls, iid, p = futs[fut]
            st = fut.result()
            stats["ok" if st in ("ok", "skip") else "fail"] += 1
            if not st.startswith("fail"):
                manifest.append({"species": cls, "photo_id": iid, "observation_id": None,
                                 "license": None, "user": None, "path": os.path.relpath(p, ROOT),
                                 "source": "biota_gobcan"})

    with open(MANIFEST, "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"HECHO {stats} · manifest={len(manifest)}")
    for cls in sorted(by):
        print(f"  {len(by[cls]):4d}  {cls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
