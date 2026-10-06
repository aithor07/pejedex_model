#!/usr/bin/env python3
"""Entrenamiento del clasificador PejeDex (PyTorch + ROCm).

Uso:
  python scripts/train.py                      # entrenamiento completo
  python scripts/train.py --smoke              # prueba rapida del pipeline
  python scripts/train.py --epochs 10          # menos epocas
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from timm.data import Mixup, create_transform, resolve_data_config
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.utils import AverageMeter, accuracy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "dataset")
CKPT = os.path.join(ROOT, "checkpoints")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mobilenetv4_conv_medium.e500_r224_in1k")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--log-every", type=int, default=50, help="pasos entre mensajes de progreso")
    p.add_argument("--save-every", type=int, default=1,
                    help="guarda checkpoints/last.pt cada N epocas (0 = solo best.pt)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--init-from", default="",
                   help="carga el backbone de otro checkpoint y reaporta la cabeza "
                        "por nombre de clase (para cuando cambia el numero de clases)")
    p.add_argument("--out", default=os.path.join(CKPT, "best.pt"))
    return p.parse_args()


def build_transforms(img_size, is_train, auto_aug="rand-m9-mstd0.5-inc1"):
    if is_train:
        t = create_transform(
            input_size=img_size, is_training=True, color_jitter=0.4,
            auto_augment=auto_aug, interpolation="bicubic",
            re_prob=0.25, re_mode="pixel", re_count=1, mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return t
    return create_transform(
        input_size=img_size, is_training=False, interpolation="bicubic",
        crop_pct=0.95, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )


def warm_start(model, path, new_classes):
    """Reutiliza un entrenamiento previo cuando cambia el numero de clases.

    Se queda con el backbone completo y reaporta la cabeza fila a fila usando
    el nombre de clase, de modo que las especies que ya estaban conservan su
    peso y las nuevas quedan con la inicializacion normal.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    old_classes = ck.get("classes") or []
    sd = dict(ck.get("model") or {})
    old_w = sd.pop("classifier.weight", None)
    old_b = sd.pop("classifier.bias", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    reused = 0
    head = model.get_classifier()
    if old_w is not None and old_classes:
        idx = {c: i for i, c in enumerate(old_classes)}
        with torch.no_grad():
            for j, c in enumerate(new_classes):
                i = idx.get(c)
                if i is None or i >= old_w.shape[0]:
                    continue
                head.weight[j].copy_(old_w[i])
                if old_b is not None and head.bias is not None:
                    head.bias[j].copy_(old_b[i])
                reused += 1
    return {"old": len(old_classes), "reused": reused,
            "nuevas": len(new_classes) - reused,
            "missing": len(missing), "unexpected": len(unexpected),
            "top1_prev": ck.get("top1"), "epoch_prev": ck.get("epoch"),
            "model_prev": ck.get("model_name")}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(DATA, "train")
    val_dir = os.path.join(DATA, "val")
    if not os.path.isdir(train_dir):
        sys.exit(f"No existe {train_dir}. Ejecuta antes scripts/build_dataset.py")

    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(args.img_size, True))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(args.img_size, False))
    classes = train_ds.classes
    assert val_ds.classes == classes, "clases distintas entre train y val"
    num_classes = len(classes)

    if args.smoke:
        train_ds = Subset(train_ds, list(range(min(400, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(120, len(val_ds)))))
        args.epochs = 1
        args.batch = min(args.batch, 64)
        args.workers = min(args.workers, 4)
        args.log_every = 5
        args.out = os.path.join(CKPT, "smoke.pt")
        print("MODO SMOKE: no toca checkpoints/best.pt", flush=True)

    counts = Counter(train_ds.targets) if hasattr(train_ds, "targets") else None
    if counts is None:
        # Subset no expone targets
        base = datasets.ImageFolder(train_dir)
        idx = train_ds.indices if isinstance(train_ds, Subset) else range(len(train_ds))
        counts = Counter(base.targets[i] for i in idx)

    print(f"clases={num_classes} train={len(train_ds):,} val={len(val_ds):,} "
          f"device={device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})")

    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                         cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                         mode="batch", label_smoothing=args.label_smoothing,
                         num_classes=num_classes)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=pin,
                            persistent_workers=args.workers > 0)

    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
    if args.init_from:
        if not os.path.exists(args.init_from):
            sys.exit(f"No existe {args.init_from}")
        info = warm_start(model, args.init_from, classes)
        print(f"warm-start desde {args.init_from}\n"
              f"  clases previas={info['old']} reutilizadas={info['reused']} "
              f"nuevas={info['nuevas']}\n"
              f"  top1 previo={info['top1_prev']} (epoca {info['epoch_prev']}) "
              f"modelo_prev={info['model_prev']}\n"
              f"  claves missing={info['missing']} unexpected={info['unexpected']}",
              flush=True)
        if info["model_prev"] and info["model_prev"] != args.model:
            print(f"  AVISO: modelo distinto ({info['model_prev']})", flush=True)
    model.to(device)

    start_epoch = 1
    best = {"top1": -1.0}
    ck = None
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch = ck.get("epoch", 0) + 1
        best = {"top1": ck.get("top1", -1.0), "macro_f1": ck.get("macro_f1", 0.0),
                "epoch": ck.get("epoch", 0)}

    total = sum(counts.values())
    w = torch.tensor([ (total / (num_classes * max(1, counts[i]))) ** 0.5
                       for i in range(num_classes)], dtype=torch.float32, device=device)
    if mixup_fn is None and args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    elif mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(weight=w)

    opt = create_optimizer_v2(model, opt="adamw", lr=args.lr, weight_decay=args.weight_decay,
                              betas=(0.9, 0.999))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, args.warmup * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    if ck is not None:
        model.load_state_dict(ck["model"])
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        if ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        print(f"reanudando desde {args.resume} (epoca {ck.get('epoch')}, "
              f"top1 {best['top1']:.2f})", flush=True)

    use_amp = (not args.no_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None
    if use_amp:
        try:
            torch.zeros(1, device=device).to(torch.bfloat16)
        except Exception:
            amp_dtype = torch.float16

    os.makedirs(CKPT, exist_ok=True)
    args_name = "args.smoke.json" if args.smoke else "args.json"
    json.dump({**vars(args), "classes": classes},
              open(os.path.join(CKPT, args_name), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    show_tr_acc = mixup_fn is None
    n_batches = len(train_loader)
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        m_top1 = AverageMeter()
        m_loss = AverageMeter()
        for it, (images, targets) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                images, targets = mixup_fn(images, targets)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            global_step += 1
            if show_tr_acc:
                acc1 = accuracy(logits, targets, topk=(1,))[0]
                m_top1.update(acc1.item(), images.size(0))
            m_loss.update(loss.item(), images.size(0))

            if it % args.log_every == 0 or it == n_batches:
                elapsed = time.time() - t0
                eta = elapsed / it * (n_batches - it)
                extra = f" top1 {m_top1.avg:.1f}" if show_tr_acc else ""
                print(f"  ep{epoch:03d} {it:5d}/{n_batches} "
                      f"loss {m_loss.avg:.4f}{extra} lr={opt.param_groups[0]['lr']:.2e} "
                      f"{elapsed:.0f}s eta {fmt_dur(eta)}", flush=True)

        # ---- validacion
        model.eval()
        v_top1, v_top5, v_loss = AverageMeter(), AverageMeter(), AverageMeter()
        crit_ce = nn.CrossEntropyLoss()
        preds_all, targs_all = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(images)
                    loss = crit_ce(logits.float(), targets)
                acc1, acc5 = accuracy(logits.float(), targets, topk=(1, min(5, num_classes)))
                v_top1.update(acc1.item(), images.size(0))
                if num_classes >= 5:
                    v_top5.update(acc5.item(), images.size(0))
                v_loss.update(loss.item(), images.size(0))
                preds_all.append(logits.float().argmax(1).cpu())
                targs_all.append(targets.cpu())

        preds = torch.cat(preds_all)
        targs = torch.cat(targs_all)
        f1 = macro_f1(targs, preds, num_classes)
        print(f"ep {epoch:03d}/{args.epochs} "
              f"loss {m_loss.avg:.4f} | val loss {v_loss.avg:.4f} "
              f"top1 {v_top1.avg:.2f} top5 {v_top5.avg:.2f} macroF1 {f1*100:.2f} "
              f"| {time.time()-t0:.0f}s lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        state = {"model": model.state_dict(), "classes": classes, "args": vars(args),
                 "epoch": epoch, "top1": v_top1.avg, "macro_f1": f1, "model_name": args.model,
                 "opt": opt.state_dict(), "sched": sched.state_dict()}
        if v_top1.avg > best["top1"]:
            best = {"top1": v_top1.avg, "macro_f1": f1, "epoch": epoch}
            torch.save(state, args.out)
            print(f"   -> nuevo mejor (top1 {v_top1.avg:.2f}) guardado en {args.out}")
        if args.save_every and epoch % args.save_every == 0:
            last = os.path.join(CKPT, "last.pt")
            tmp = last + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, last)   # atomico: nunca queda un last.pt a medias
            if best.get("epoch") != epoch:
                print(f"   -> progreso guardado en {last} (epoca {epoch})", flush=True)

    print(f"\nMEJOR top1={best['top1']:.2f} macroF1={best.get('macro_f1',0)*100:.2f} "
          f"epoca={best['epoch']}")
    json.dump(best, open(os.path.join(CKPT, "best_metrics.json"), "w"), indent=1)
    return 0


def fmt_dur(sec):
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def macro_f1(targs, preds, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (targs == c)).sum())
        fp = int(((preds == c) & (targs != c)).sum())
        fn = int(((preds != c) & (targs == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s)


if __name__ == "__main__":
    sys.exit(main())
