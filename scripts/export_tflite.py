#!/usr/bin/env python3
"""Exporta el checkpoint PyTorch de PejeDex a TFLite (float32 e INT8).

Pipeline: PyTorch -> ONNX -> SavedModel (onnx2tf) -> TFLite (+ cuantizacion PTQ).
La normalizacion (x/255 - mean)/std va DENTRO del grafo, de modo que el modelo
TFLite acepta directamente el pixel RGB en bruto (uint8 0..255).

Uso:
  python scripts/export_tflite.py                     # usa checkpoints/best.pt
  python scripts/export_tflite.py --checkpoint x.pt
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(ROOT, "checkpoints")
MODEL_DIR = os.path.join(ROOT, "model")
BUILD = os.path.join(ROOT, "build")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG = 224


def get_torch():
    import torch
    return torch


def make_deploy_model(ckpt_path):
    import timm
    import torch
    import torch.nn as nn

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    name = ck.get("model_name") or ck["args"].get("model")
    core = timm.create_model(name, pretrained=False, num_classes=len(classes))
    core.load_state_dict(ck["model"])
    core.eval()

    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)

    class Deploy(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        def forward(self, x):          # x: float32 0..255, NCHW
            x = x * (1.0 / 255.0)
            x = (x - self.mean) / self.std
            return self.core(x)

    return Deploy().eval(), classes, name


def export_onnx(model, path):
    torch = get_torch()
    dummy = torch.zeros(1, 3, IMG, IMG)
    torch.onnx.export(
        model, dummy, path, input_names=["input"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamic_axes=None,
        dynamo=False,
    )
    return path


def to_savedmodel(onnx_path, out_dir):
    import onnx2tf
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        batch_size=1,
        non_verbose=True,
        disable_group_convolution=True,
        tflite_backend="tf_converter",   # produce SavedModel; el default
                                         # 'flatbuffer_direct' no cuantiza bien
    )
    return out_dir


def preprocess(im, mode="crop"):
    """PIL RGB -> numpy uint8 224x224x3.

    mode="crop" replica create_transform(is_training=False, crop_pct=0.95):
    redimensiona el lado corto a 224/0.95=236 y recorta 224 por el centro.
    mode="resize" es el resize simple que haria la app.
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


def rep_dataset(paths, n=200):
    from PIL import Image

    sample = []
    rng = np.random.default_rng(0)
    sel = paths if len(paths) <= n else list(rng.choice(paths, n, replace=False))
    for p in sel:
        try:
            sample.append(preprocess(Image.open(p).convert("RGB"), "crop"))
        except Exception:
            continue
    if not sample:
        raise RuntimeError("sin imagenes para el dataset representativo")
    return sample


def convert_tflite(saved_dir, rep, kind, out_path):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    if kind == "float32":
        pass
    elif kind == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif kind == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        # calibracion: el SavedModel de entrada es float32 (la conversion a
        # uint8 en inference_input_type ocurre despues), así que hay que
        # ceder float32 aunque preprocess() devuelva uint8.
        conv.representative_dataset = lambda: iter(
            [[a.astype(np.float32)] for a in rep])
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
        conv.inference_output_type = tf.float32
    else:
        raise ValueError(kind)
    buf = conv.convert()
    with open(out_path, "wb") as f:
        f.write(buf)
    return len(buf)


def eval_tflite(path, val_dir, classes, n=2000, use_uint8=True, mode="crop"):
    import random
    import tensorflow as tf
    from PIL import Image

    files = []
    for c in classes:
        d = os.path.join(val_dir, c.replace(" ", "_"))
        if os.path.isdir(d):
            files += [(os.path.join(d, f), i) for i, f in enumerate(sorted(os.listdir(d)))
                      if f.endswith((".jpg", ".jpeg", ".png"))]
    rng = random.Random(0)
    rng.shuffle(files)
    files = files[:n]
    if not files:
        return None

    itp = tf.lite.Interpreter(model_path=path, num_threads=4)
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]
    name_to_idx = {c: i for i, c in enumerate(classes)}

    top1 = 0
    for p, _ in files:
        cls = os.path.basename(os.path.dirname(p))
        try:
            a = preprocess(Image.open(p).convert("RGB"), mode)
        except Exception:
            continue
        if inp["dtype"] == np.uint8:
            x = a[np.newaxis, ...].astype(np.uint8)
        else:
            x = a[np.newaxis, ...].astype(np.float32)
        itp.set_tensor(inp["index"], x)
        itp.invoke()
        y = itp.get_tensor(out["index"])[0]
        if int(np.argmax(y)) == name_to_idx[cls]:
            top1 += 1
    return top1 / max(1, len(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(CKPT, "best.pt"))
    ap.add_argument("--kinds", default="float32,dynamic,int8")
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--rep-dir", default=os.path.join(ROOT, "dataset", "val"))
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"No existe {args.checkpoint}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)

    print("cargando checkpoint...")
    model, classes, name = make_deploy_model(args.checkpoint)
    print(f"  modelo={name} clases={len(classes)}")

    onnx_path = os.path.join(BUILD, "pejedex.onnx")
    print("exportando ONNX...")
    export_onnx(model, onnx_path)
    print(f"  {os.path.getsize(onnx_path)/1e6:.1f} MB")

    print("onnx2tf -> SavedModel...")
    saved = to_savedmodel(onnx_path, os.path.join(BUILD, "saved_model"))

    rep_paths = []
    if os.path.isdir(args.rep_dir):
        for dp, _dn, fn in os.walk(args.rep_dir):
            rep_paths += [os.path.join(dp, f) for f in fn if f.endswith((".jpg", ".jpeg", ".png"))]
    rep = rep_dataset(rep_paths) if rep_paths else None

    val_dir = os.path.join(ROOT, "dataset", "val")
    results = {}
    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        out = os.path.join(MODEL_DIR, f"pejedex_{kind}.tflite")
        print(f"convirtiendo {kind}...")
        t0 = time.time()
        try:
            sz = convert_tflite(saved, rep, kind, out)
        except Exception as e:  # noqa: BLE001
            print(f"  FALLO {kind}: {e}")
            results[kind] = {"error": str(e)}
            continue
        acc_crop = acc_resize = None
        if os.path.isdir(val_dir) and rep is not None:
            acc_crop = eval_tflite(out, val_dir, classes, n=args.eval_n, mode="crop")
            acc_resize = eval_tflite(out, val_dir, classes, n=args.eval_n, mode="resize")
        results[kind] = {"bytes": sz, "mb": round(sz / 1e6, 2),
                         "top1": None if acc_crop is None else round(acc_crop * 100, 2),
                         "top1_resize": None if acc_resize is None else round(acc_resize * 100, 2),
                         "seconds": round(time.time() - t0, 1)}
        print(f"  {sz/1e6:6.2f} MB  top1={results[kind]['top1']} "
              f"(resize {results[kind]['top1_resize']})  ({results[kind]['seconds']}s)")

    # elegir el mejor que cumpla 30 MB
    valid = {k: v for k, v in results.items() if "error" not in v and v["mb"] <= 30}
    pick = None
    if valid:
        with_acc = {k: v for k, v in valid.items() if v["top1"] is not None}
        pick = max(with_acc or valid, key=lambda k: (valid[k]["top1"] or -1))
        shutil.copy(os.path.join(MODEL_DIR, f"pejedex_{pick}.tflite"),
                    os.path.join(MODEL_DIR, "pejedex.tflite"))

    with open(os.path.join(MODEL_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c.replace("_", " ") + "\n")

    meta = {"model": name, "num_classes": len(classes), "img_size": IMG,
            "input": "uint8 [1,224,224,3] RGB en bruto (0..255)" if pick == "int8"
                     else "float32 [1,224,224,3] RGB 0..255",
            "normalize": "dentro del grafo: (x/255 - mean)/std con ImageNet mean/std",
            "mean": list(MEAN), "std": list(STD),
            "output": "float32 [1,N] logits",
            "results": results, "chosen": pick, "classes": classes}
    json.dump(meta, open(os.path.join(MODEL_DIR, "export_metadata.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\nRESUMEN")
    for k, v in results.items():
        print(f"  {k:9s} {v}")
    print(f"elegido: {pick}")
    if os.path.exists(os.path.join(MODEL_DIR, "pejedex.tflite")):
        print(f"-> {os.path.join(MODEL_DIR, 'pejedex.tflite')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
