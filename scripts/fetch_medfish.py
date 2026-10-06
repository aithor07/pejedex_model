#!/usr/bin/env python3
"""Descarga MEDFISH101 (69.153 enlaces iNaturalist, 101 especies, CC).

Entrada : CSV con columnas image_id, class_name, image_link (tamano 'medium').
Salida  : data_raw/medfish/<Especie>/<photo_id>.jpg
          data_raw/medfish_manifest.jsonl
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "PejeDex-dataset/1.0 (biodiversity training dataset)"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data_raw")
CSV_PATH = os.path.join(RAW, "medfish_links.csv")
SRC_CSV = "/tmp/medfish.csv"
DIR = os.path.join(RAW, "medfish")
MANIFEST = os.path.join(RAW, "medfish_manifest.jsonl")

SIZES = ("large", "medium")


def upgrade(url, size):
    m = re.match(r"^(.*)/([a-zA-Z]+)\.(jpe?g|png)$", url)
    if not m:
        return url
    stem, _sz, ext = m.groups()
    return f"{stem}/{size}.{ext}"


def download(url, path, timeout=60):
    if os.path.exists(path) and os.path.getsize(path) > 4096:
        return "skip"
    last = None
    for sz in SIZES:
        try:
            req = urllib.request.Request(upgrade(url, sz), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) < 4096:
                continue
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            return "ok"
        except Exception as e:  # noqa: BLE001
            last = e
    return f"fail:{last}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit-per-class", type=int, default=1000)
    ap.add_argument("--classes", default="", help="csv de especies; vacio = todas")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        os.replace(SRC_CSV, CSV_PATH) if os.path.exists(SRC_CSV) else None
    if not os.path.exists(CSV_PATH):
        sys.exit(f" falta {CSV_PATH}")

    allowed = {c.strip() for c in args.classes.split(",") if c.strip()} or None
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    if allowed:
        rows = [r for r in rows if r["class_name"] in allowed]

    per_class = {}
    keep = []
    for r in rows:
        cls = r["class_name"]
        if per_class.get(cls, 0) >= args.limit_per_class:
            continue
        per_class[cls] = per_class.get(cls, 0) + 1
        keep.append(r)

    print(f"descargando {len(keep)} imagenes / {len(per_class)} clases", flush=True)

    manifest = []
    if os.path.exists(MANIFEST):
        manifest = [json.loads(l) for l in open(MANIFEST, encoding="utf-8") if l.strip()]
    have = {m["photo_id"] for m in manifest}
    todo = [r for r in keep if r["image_id"] not in have]
    print(f"  en manifiesto {len(have)}, pendientes {len(todo)}", flush=True)

    def job(r):
        cls_dir = os.path.join(DIR, r["class_name"].replace(" ", "_"))
        os.makedirs(cls_dir, exist_ok=True)
        path = os.path.join(cls_dir, f"{r['image_id']}.jpg")
        return r, download(r["image_link"], path), os.path.relpath(path, ROOT)

    stats = {"ok": 0, "skip": 0, "fail": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(job, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r, st, rel = fut.result()
            stats["ok" if st in ("ok", "skip") else "fail"] += 1
            if not st.startswith("fail"):
                have.add(r["image_id"])
                manifest.append({"species": r["class_name"], "photo_id": r["image_id"],
                                 "observation_id": None, "license": None, "user": None,
                                 "path": rel, "source": "medfish101"})
            if i % 500 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(todo)} ok={stats['ok']} fail={stats['fail']} "
                      f"{i/el:.1f} img/s", flush=True)
                with open(MANIFEST, "w", encoding="utf-8") as f:
                    for m in manifest:
                        f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with open(MANIFEST, "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"HECHO {stats} manifest={len(manifest)} en {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
