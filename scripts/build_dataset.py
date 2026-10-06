#!/usr/bin/env python3
"""Construye dataset/train y dataset/val a partir de los manifiestos.

 - valida cada imagen (legible, min 200 px de lado corto)
 - elimina duplicados exactos (md5) y casi-duplicados (aHash+dHash)
 - agrupa casi-duplicados en clusters y reparte los clusters entre train/val
   para que ningun mismo pez/observacion aparezca en los dos lados
 - descarta clases por debajo de --min-per-class
 - crea enlaces simbolicos (o copias) en dataset/<split>/<Clase>/
 - escribe dataset/classes.json y dataset/split.json

Salida adicional: dataset/stats.json con el recuento final por clase.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict

from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data_raw")
OUT = os.path.join(ROOT, "dataset")

MANIFESTS = ["medfish_manifest.jsonl", "inat_manifest.jsonl", "biota_manifest.jsonl"]


def load_manifests():
    recs = []
    for name in MANIFESTS:
        p = os.path.join(RAW, name)
        if not os.path.exists(p):
            print(f"  (falta {name})")
            continue
        n0 = len(recs)
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        print(f"  {name}: {len(recs) - n0}")

    # ficheros huérfanos: existen en disco pero no estan en ningun manifiesto
    referenced = {r["path"] for r in recs}
    for root, src in (("medfish", "medfish101"), ("inat", "inaturalist"), ("biota", "biota_gobcan")):
        base = os.path.join(RAW, root)
        if not os.path.isdir(base):
            continue
        n0 = len(recs)
        for cls_folder in sorted(os.listdir(base)):
            d = os.path.join(base, cls_folder)
            if not os.path.isdir(d):
                continue
            cls = cls_folder.replace("_", " ")
            for f in sorted(os.listdir(d)):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                rel = os.path.relpath(os.path.join(d, f), ROOT)
                if rel in referenced:
                    continue
                referenced.add(rel)
                recs.append({"species": cls, "photo_id": os.path.splitext(f)[0],
                             "observation_id": None, "license": None, "user": None,
                             "path": rel, "source": src})
        if len(recs) != n0:
            print(f"  +{len(recs)-n0} huérfanos de {root}")
    return recs


def ahash(im):
    g = ImageOps.grayscale(im).resize((8, 8), Image.LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / 64
    v = 0
    for i, p in enumerate(px):
        if p > avg:
            v |= 1 << i
    return v


def dhash(im):
    g = ImageOps.grayscale(im).resize((9, 8), Image.LANCZOS)
    px = list(g.getdata())
    v = 0
    for r in range(8):
        for c in range(8):
            left = px[r * 9 + c]
            right = px[r * 9 + c + 1]
            v = (v << 1) | (1 if left > right else 0)
    return v


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def hamming(a, b):
    return (a ^ b).bit_count()


def validate_one(r):
    """Valida y hashea una imagen. Ejecutable en paralelo."""
    path = os.path.join(ROOT, r["path"])
    try:
        if not os.path.exists(path):
            return {"error": "missing"}
        if os.path.getsize(path) < 8000:
            return {"error": "tiny_file"}
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            if min(w, h) < 200:
                return {"error": "low_res"}
            im = im.convert("RGB")
            digest = hashlib.md5(im.tobytes()).hexdigest()
            ah, dh = ahash(im), dhash(im)
        return {"rec": r, "path": path, "md5": digest, "ah": ah, "dh": dh, "w": w, "h": h}
    except Exception:  # noqa: BLE001
        return {"error": "corrupt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-per-class", type=int, default=100)
    ap.add_argument("--keep-unnamed", action="store_true",
                    help="conserva clases sin nombre en español (por defecto se descartan)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--copy", action="store_true", help="copiar en vez de enlazar")
    ap.add_argument("--hash-dist", type=int, default=6, help="distancia maxima de aHash para duplicado")
    ap.add_argument("--sample", type=int, default=0,
                    help="probar con N registros aleatorios (0 = todo)")
    ap.add_argument("--out", default=OUT, help="carpeta de salida")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("cargando manifiestos...")
    recs = load_manifests()
    if args.sample:
        import random
        random.Random(0).shuffle(recs)
        recs = recs[:args.sample]
        print(f"muestra de prueba: {len(recs)} registros")
    print(f"total registros: {len(recs)}")

    # 0) descartar clases sin nombre en español (antes de validar para no
    #    abrir imagenes que van a salir)
    names_es = {}
    p_names = os.path.join(RAW, "names_es.json")
    if os.path.exists(p_names):
        names_es = json.load(open(p_names, encoding="utf-8"))
    if names_es and not args.keep_unnamed:
        present = {r["species"] for r in recs}
        unnamed = sorted(c for c in present if not str(names_es.get(c, "")).strip())
        if unnamed:
            n0 = len(recs)
            recs = [r for r in recs
                    if str(names_es.get(r["species"], "")).strip()]
            print(f"sin nombre ES, descartadas {len(unnamed)} clases / "
                  f"{n0 - len(recs)} registros: "
                  f"{', '.join(unnamed[:6])}{' ...' if len(unnamed) > 6 else ''}")

    # 1) validar imagenes (paralelo)
    from concurrent.futures import ProcessPoolExecutor
    workers = min(32, os.cpu_count() or 4)
    print(f"validando imagenes con {workers} procesos...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(validate_one, recs, chunksize=64), 1):
            results.append(res)
            if i % 10000 == 0:
                print(f"  {i}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)

    seen_md5 = set()
    valid = []
    bad = Counter()
    for res in results:
        if res.get("error"):
            bad[res["error"]] += 1
            continue
        if res["md5"] in seen_md5:
            bad["dup_exact"] += 1
            continue
        seen_md5.add(res["md5"])
        r = res["rec"]
        valid.append({**r, "_path": res["path"], "_ah": res["ah"], "_dh": res["dh"],
                      "_w": res["w"], "_h": res["h"]})
    del results
    print(f"validas={len(valid)} descartadas={dict(bad)} ({time.time()-t0:.0f}s)")

    # 2) clusters por clase (casi-duplicados) + opcionmente por observacion
    print("agrupando casi-duplicados por clase...")
    by_class = defaultdict(list)
    for r in valid:
        by_class[r["species"]].append(r)

    clusters = {}   # class -> list of [records]
    for cls, items in by_class.items():
        n = len(items)
        dsu = DSU(n)
        for i in range(n):
            for j in range(i + 1, n):
                if dsu.find(i) == dsu.find(j):
                    continue
                if hamming(items[i]["_ah"], items[j]["_ah"]) <= args.hash_dist and \
                   hamming(items[i]["_dh"], items[j]["_dh"]) <= 8:
                    dsu.union(i, j)
        # toda foto de la misma observacion de iNat queda en el mismo lado
        by_obs = defaultdict(list)
        for i, r in enumerate(items):
            if r.get("observation_id"):
                by_obs[r["observation_id"]].append(i)
        for idxs in by_obs.values():
            for j in idxs[1:]:
                dsu.union(idxs[0], j)
        groups = defaultdict(list)
        for i in range(n):
            groups[dsu.find(i)].append(items[i])
        clusters[cls] = list(groups.values())

    # 3) descartar clases pequeñas
    counts = {cls: sum(len(g) for g in gs) for cls, gs in clusters.items()}
    keep = {c for c, n in counts.items() if n >= args.min_per_class}
    dropped = {c: counts[c] for c in counts if c not in keep}
    print(f"clases: {len(counts)} · se conservan {len(keep)} · descartadas {len(dropped)}")
    for c, n in sorted(dropped.items(), key=lambda x: -x[1]):
        print(f"   {n:4d}  {c}")

    # 4) repartir clusters train/val (de menor a mayor: val se llena con
    #    clusters pequenos y los grandes se quedan en train)
    print("repartiendo train/val...")
    split = {"train": {}, "val": {}}
    for cls in sorted(keep):
        gs = sorted(clusters[cls], key=lambda g: len(g))
        total = counts[cls]
        target = total * args.val_frac
        if len(gs) == 1:
            # todo casi-duplicado: reparto por img (habra algo de fuga)
            g = gs[0]
            k = max(1, int(len(g) * args.val_frac))
            split["val"][cls] = g[:k]
            split["train"][cls] = g[k:] or g
            continue
        train, val, val_imgs = [], [], 0
        for g in gs:
            if val_imgs < target:
                val.extend(g)
                val_imgs += len(g)
            else:
                train.extend(g)
        # garantizar clase en ambos lados
        if not val:
            val.extend(gs[0])
        if not train:
            train.extend(gs[-1])
            val = [g for g in val if g is not gs[-1]] or val
        split["train"][cls] = train
        split["val"][cls] = val

    # 5) crear enlaces/copias
    print("materializando...")
    if os.path.abspath(args.out) == os.path.abspath(ROOT):
        sys.exit("--out no puede ser la raiz del proyecto")
    for sp in ("train", "val"):
        stale = os.path.join(args.out, sp)
        if os.path.isdir(stale):
            shutil.rmtree(stale)   # limpia enlaces de la corrida anterior
    for sp in ("train", "val"):
        for cls, items in split[sp].items():
            d = os.path.join(args.out, sp, cls.replace(" ", "_"))
            os.makedirs(d, exist_ok=True)
            for r in items:
                dst = os.path.join(d, os.path.basename(r["path"]))
                if os.path.lexists(dst):
                    os.remove(dst)
                if args.copy:
                    shutil.copy2(r["_path"], dst)
                else:
                    os.symlink(os.path.relpath(r["_path"], d), dst)

    # 6) salidas
    classes = sorted(keep)
    json.dump({"classes": classes,
               "common_es": {c: names_es.get(c, "") for c in classes}},
              open(os.path.join(args.out, "classes.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    stats = {}
    for sp in ("train", "val"):
        stats[sp] = {cls: len(v) for cls, v in split[sp].items()}
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    tr = sum(stats["train"].values())
    va = sum(stats["val"].values())
    print(f"\nCLASES {len(classes)}")
    print(f"train {tr:,}  val {va:,}  total {tr+va:,}")
    print(f"min/clase train={min(stats['train'].values())} max={max(stats['train'].values())}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
