#!/usr/bin/env python3
"""Descarga masiva de fotos con licencia abierta desde iNaturalist.

Modos:
  probe  -> cuenta fotos CC disponibles por especie (1 peticion por especie)
  fetch  -> descarga hasta --max fotos por especie (con reanudacion)

Los ficheros se guardan como  <binomial>/<photo_id>.jpg
Se guarda un manifiesto con observation_id (para separar train/val sin fugas)
y con autor/licencia/atribucion.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://api.inaturalist.org/v1"
UA = "PejeDex-dataset/1.0 (biodiversity training dataset; contact: local)"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data_raw")
DIR = os.path.join(RAW, "inat")
PROBE = os.path.join(RAW, "inat_probe.json")
MANIFEST = os.path.join(RAW, "inat_manifest.jsonl")

OPEN_LICENSES = "cc-by,cc-by-sa,cc0,pdm"
SIZES = ("original", "large", "medium", "square")


def get(url, tries=8, timeout=60):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            # 429/5xx y cortes de conexion: espera mas y con algo de ruido
            time.sleep(min(20.0, 1.5 * (2 ** i)) + random.random())
    raise RuntimeError(f"GET failed {url}: {last}")


def candidates():
    with open(os.path.join(HERE, "species_candidates.json"), encoding="utf-8") as f:
        return json.load(f)["candidates"]


def _exact(results, name):
    """Primero igualdad exacta, si no el mejor del mismo genero."""
    best = None
    for t in results:
        if t.get("name", "").lower() == name.lower() and t.get("rank") == "species":
            return t["id"], t.get("name"), t.get("preferred_common_name")
        if best is None and name.lower().split()[0] == t.get("name", "").lower().split(" ")[0]:
            best = t
    if best is not None:
        return best["id"], best.get("name"), best.get("preferred_common_name")
    return None


def resolve_taxon(name):
    """Taxon id exacto para un binomial (o el mejor parecido).

    /taxa ordena por numero de observaciones, así que un binomial poco observado
    (p.ej. Sarda sarda, 436 obs) no llega al per_page=8 y se quedaba sin resolver.
    Se intenta /taxa/autocomplete, que ordena por coincidencia de nombre.
    """
    url = f"{API}/taxa?" + urllib.parse.urlencode({"q": name, "rank": "species", "per_page": 8})
    hit = _exact(get(url).get("results", []), name)
    if hit:
        return hit

    url = f"{API}/taxa/autocomplete?" + urllib.parse.urlencode({"q": name, "per_page": 10})
    hit = _exact(get(url).get("results", []), name)
    if hit:
        return hit
    return None, None, None


def count_photos(taxon_id, research=True):
    p = {
        "taxon_id": taxon_id, "photos": "true", "photo_license": OPEN_LICENSES, "per_page": 1,
    }
    if research:
        p["quality_grade"] = "research"
    return get(f"{API}/observations?" + urllib.parse.urlencode(p))["total_results"]


def iter_photos(taxon_id, research=True, limit=1000):
    """Genera dicts de foto ordenados por votos (calidad visual aproximada)."""
    page, got = 1, 0
    while got < limit:
        p = {
            "taxon_id": taxon_id, "photos": "true", "photo_license": OPEN_LICENSES,
            "per_page": 200, "page": page, "order_by": "votes", "order": "desc",
        }
        if research:
            p["quality_grade"] = "research"
        d = get(f"{API}/observations?" + urllib.parse.urlencode(p))
        res = d.get("results", [])
        if not res:
            break
        for o in res:
            oid = o.get("id")
            uname = (o.get("user") or {}).get("login")
            for ph in o.get("photos", []) or []:
                yield {
                    "photo_id": ph.get("id"),
                    "observation_id": oid,
                    "url": ph.get("url"),
                    "license": ph.get("license_code"),
                    "user": uname,
                    "sfw": ph.get("square_url"),
                }
                got += 1
                if got >= limit:
                    return
        if page * 200 >= d.get("total_results", 0):
            break
        page += 1
        time.sleep(0.2)


def sized_urls(url):
    if not url:
        return []
    # https://.../photos/<id>/square.jpeg -> cambiar sufijo de tamano
    m = re.match(r"^(.*)/([a-z]+)\.(jpe?g|png)$", url)
    if not m:
        return [url]
    stem, _size, ext = m.groups()
    return [f"{stem}/{s}.{ext}" for s in SIZES] + [url]


def download_one(url, path, timeout=60):
    if os.path.exists(path) and os.path.getsize(path) > 4096:
        return "skip"
    last = None
    for cand in sized_urls(url):
        try:
            req = urllib.request.Request(cand, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) < 4096:
                continue
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return "ok"
        except Exception as e:  # noqa: BLE001
            last = e
    return f"fail:{last}"


def cmd_probe(args):
    out = {}
    if os.path.exists(PROBE):
        out = json.load(open(PROBE, encoding="utf-8"))
    for i, c in enumerate(candidates(), 1):
        key = c["key"]
        if key in out and out[key].get("taxon_id"):
            continue
        try:
            tid, sci, common = resolve_taxon(key)
            n_res = count_photos(tid, True) if tid else 0
            n_any = count_photos(tid, False) if tid else 0
            out[key] = {"taxon_id": tid, "resolved": sci, "common": common,
                        "cc_research": n_res, "cc_any": n_any}
            print(f"[{i}/{len(candidates())}] {key:32s} id={tid} research={n_res} any={n_any}", flush=True)
        except Exception as e:  # noqa: BLE001
            out[key] = {"taxon_id": None, "error": str(e)}
            print(f"[{i}] {key} ERROR {e}", flush=True)
        with open(PROBE, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        time.sleep(0.35)
    print("\nRESUMEN")
    ok = [v for v in out.values() if v.get("taxon_id")]
    print(f"  especies resueltas: {len(ok)}/{len(out)}")
    print(f"  suma cc_research: {sum(v['cc_research'] for v in ok)}")
    print(f"  suma cc_any:      {sum(v['cc_any'] for v in ok)}")
    return 0


def cmd_fetch(args):
    os.makedirs(DIR, exist_ok=True)
    probe = json.load(open(PROBE, encoding="utf-8"))
    skip = set()
    if args.skip_csv and os.path.exists(args.skip_csv):
        import csv as _csv
        with open(args.skip_csv, encoding="utf-8") as f:
            skip = {r["class_name"] for r in _csv.DictReader(f)}
        print(f"excluyendo {len(skip)} especies ya cubiertas por {args.skip_csv}", flush=True)
    manifest = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            manifest = [json.loads(l) for l in f if l.strip()]
    have = {m["photo_id"] for m in manifest}
    by_sp = {}
    for m in manifest:
        by_sp[m["species"]] = by_sp.get(m["species"], 0) + 1

    for c in candidates():
        key = c["key"]
        info = probe.get(key) or {}
        tid = info.get("taxon_id")
        if not tid:
            print(f"SKIP {key} (sin taxon)", flush=True)
            continue
        if key in skip:
            continue
        if info.get("cc_research", 0) < args.min_count:
            continue
        need = args.max - by_sp.get(key, 0)
        if need <= 0:
            continue
        folder = os.path.join(DIR, key.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)

        # un fallo de red en una especie no debe matar el resto de la cola
        try:
            got_new, local = [], set()
            for grade in (True, False):
                if len(got_new) >= need:
                    break
                if not grade and info.get("cc_any", 0) > info.get("cc_research", 0):
                    print(f"  {key}: rellenando con quality_grade=any", flush=True)
                for ph in iter_photos(tid, research=grade, limit=need + 400):
                    pid = ph["photo_id"]
                    if pid in have or pid in local:
                        continue
                    if len(got_new) >= need:
                        break
                    local.add(pid)
                    got_new.append(ph)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {key}: {e}", flush=True)
            continue

        if not got_new:
            print(f"0 nuevas para {key} (tenemos {by_sp.get(key,0)})", flush=True)
            continue

        jobs = []
        for ph in got_new:
            path = os.path.join(folder, f"{ph['photo_id']}.jpg")
            jobs.append((ph, path))
        results = {"ok": 0, "skip": 0, "fail": 0}
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(download_one, ph["url"], path): ph for ph, path in jobs}
            for fut in as_completed(futs):
                ph = futs[fut]
                st = fut.result()
                results["ok" if st in ("ok", "skip") else "fail"] += 1
                if st.startswith("fail"):
                    continue
                have.add(ph["photo_id"])
                manifest.append({
                    "species": key, "photo_id": ph["photo_id"], "observation_id": ph["observation_id"],
                    "license": ph["license"], "user": ph["user"],
                    "path": os.path.relpath(os.path.join(DIR, key.replace(" ", "_"),
                                                         f"{ph['photo_id']}.jpg"), ROOT),
                })
        by_sp[key] = by_sp.get(key, 0) + results["ok"] + results["skip"]
        print(f"{key:32s} nuevas={results['ok']} fail={results['fail']} total={by_sp[key]}", flush=True)
        with open(MANIFEST, "w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"\nTOTAL manifest={len(manifest)}")
    return 0


def main():
    global OPEN_LICENSES, PROBE
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "fetch"])
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--skip-csv", default="", help="CSV con columna class_name a excluir")
    ap.add_argument("--min-count", type=int, default=40,
                    help="ignora especies con menos fotos CC de esta cantidad")
    ap.add_argument("--licenses", default=OPEN_LICENSES,
                    help="lista de licencias de foto para iNaturalist")
    ap.add_argument("--probe", default=PROBE, help="fichero de recuentos")
    args = ap.parse_args()
    OPEN_LICENSES = args.licenses
    PROBE = args.probe
    return cmd_probe(args) if args.mode == "probe" else cmd_fetch(args)


if __name__ == "__main__":
    sys.exit(main())
