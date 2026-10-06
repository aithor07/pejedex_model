#!/usr/bin/env python3
"""Indexa la galeria BIOTA del Gobierno de Canarias (17.727 imagenes).

GET /biota/galeria/imagenes?lang=es&currentImage=N&pageSize=M  -> trozo de HTML con <img>.
Salida: data_raw/biota_index.json  -> [{species_code, image_id, scientific, author_hint}]
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://www.biodiversidadcanarias.es"
API = BASE + "/biota/galeria/imagenes"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
OUT = os.path.join(os.path.dirname(__file__), "..", "data_raw", "biota_index.json")

ITEM = re.compile(
    r'<a href="/biota/especie/(?P<sp>[A-Z]\d+)"[^>]*>.*?'
    r'<img alt="(?P<alt>[^"]*)" src="/biota/especie/(?P=sp)/imagenes/(?P<im>\d+)',
    re.S,
)


def get(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "es"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def clean_alt(alt):
    # "<i>Diplodus sargus</i> (Linnaeus, 1758)" -> "Diplodus sargus"
    t = re.sub(r"<[^>]+>", "", alt).strip()
    m = re.match(r"^([A-Z][A-Za-z]+ [a-z][a-z\-]+)", t)
    return (m.group(1) if m else t), t


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    items, seen, offset, empty = [], set(), 0, 0
    PAGE = 500
    while True:
        url = f"{API}?lang=es&currentImage={offset}&pageSize={PAGE}"
        html = get(url)
        found = list(ITEM.finditer(html))
        if not found:
            empty += 1
            if empty >= 2:
                break
            offset += PAGE
            continue
        empty = 0
        for m in found:
            sp, im = m.group("sp"), m.group("im")
            k = (sp, im)
            if k in seen:
                continue
            seen.add(k)
            sci, full = clean_alt(m.group("alt"))
            items.append({"species_code": sp, "image_id": im, "scientific": sci, "full": full})
        print(f"offset={offset} match={len(found)} total={len(items)}", flush=True)
        if len(found) < PAGE:
            break
        offset += PAGE
        time.sleep(0.25)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    from collections import Counter
    c = Counter(i["scientific"] for i in items)
    print(f"\nINDEXADOS {len(items)} imagenes / {len(c)} especies -> {OUT}")
    for name, n in c.most_common(40):
        print(f"  {n:5d}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
