#!/usr/bin/env python3
"""Clasifica imagenes con el modelo TFLite de PejeDex.

Uso:
  python scripts/predict.py lubina.jpeg
  python scripts/predict.py lubina.jpeg mero.jpeg
  python scripts/predict.py --topk 3 --mode resize *.jpeg
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # calla avisos de TF/CUDA

import argparse
import contextlib
import json
import time
import warnings

import numpy as np


@contextlib.contextmanager
def _calla_stderr():
    """Redirige el fd 2 a /dev/null mientras se crea el interpreter.

    Los avisos de TF/absl/XNNPACK los escribe el C++ directamente en stderr
    antes de inicializar, asi que no los alcanza TF_CPP_MIN_LOG_LEVEL.
    Las excepciones de Python se propagan *despues* de restaurar el fd.
    """
    sys.stderr.flush()
    guardado = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(guardado, 2)
        os.close(guardado)
        os.close(devnull)


with _calla_stderr():
    import tensorflow as tf
    from PIL import Image

warnings.filterwarnings("ignore", message=r".*tf\.lite\.Interpreter is deprecated.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "model", "pejedex.tflite")
CLASSES_JSON = os.path.join(ROOT, "dataset", "classes.json")
IMG = 224


def preprocess(im, mode="crop"):
    """crop = igual que el entrenamiento (lado corto a 224/0.95=236 + recorte
    central). resize = resize simple, como haria una app sin mas.
    """
    if mode == "crop":
        target = int(round(IMG / 0.95))
        w, h = im.size
        scale = target / min(w, h)
        nw, nh = max(IMG, int(round(w * scale))), max(IMG, int(round(h * scale)))
        im = im.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - IMG) // 2, (nh - IMG) // 2
        im = im.crop((left, top, left + IMG, top + IMG))
    else:
        im = im.resize((IMG, IMG), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description="Clasifica una o varias fotos con model/pejedex.tflite",
        epilog="ejemplo: scripts/predict.py lubina.jpeg")
    ap.add_argument("imagenes", nargs="*", help="ruta(s) a la(s) foto(s)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--mode", choices=["crop", "resize"], default="crop")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if not args.imagenes:
        raiz = sorted(f for f in os.listdir(ROOT)
                      if f.lower().endswith((".jpg", ".jpeg", ".png")))
        print("indica al menos una imagen.\n")
        print("uso: scripts/predict.py <foto>")
        if raiz:
            print("\nfotos en la raiz del proyecto:")
            for f in raiz:
                print(f"  scripts/predict.py {f}")
        return 2

    if not os.path.exists(args.model):
        sys.exit(f"no existe {args.model}")

    labels = [l.rstrip("\n") for l in
              open(os.path.join(ROOT, "model", "labels.txt"), encoding="utf-8")]
    common = {}
    if os.path.exists(CLASSES_JSON):
        common = json.load(open(CLASSES_JSON, encoding="utf-8")).get("common_es", {})

    with _calla_stderr():
        itp = tf.lite.Interpreter(model_path=args.model, num_threads=args.threads)
        itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]

    for path in args.imagenes:
        if not os.path.exists(path):
            print(f"{path}: no existe\n")
            continue
        a = preprocess(Image.open(path).convert("RGB"), args.mode)
        x = a[np.newaxis, ...].astype(inp["dtype"])
        t = time.perf_counter()
        itp.set_tensor(inp["index"], x)
        itp.invoke()
        y = itp.get_tensor(out["index"])[0].astype(np.float64)
        ms = (time.perf_counter() - t) * 1000

        e = np.exp(y - y.max())
        p = e / e.sum()
        order = np.argsort(-p)[:args.topk]

        print(f"{os.path.basename(path)}  [{ms:.1f} ms, entrada {inp['dtype'].__name__}]")
        for rank, i in enumerate(order, 1):
            sci = labels[i]
            es = common.get(sci) or "-"
            bar = "#" * int(p[i] * 40)
            print(f"  {rank}. {p[i]*100:5.1f}% {bar:<40s} {es}  ({sci})")
        top = order[0]
        print(f"  => {common.get(labels[top]) or '-'} ({labels[top]})  {p[top]*100:.1f}%")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
