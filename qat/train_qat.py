"""Brevitas QAT fine-tuning for YOLOv5n(ReLU). Phase 4.

Target is W4A8: the bit-width sweep (qat/ptq_baseline.py) showed INT8 needs no
fine-tuning at all (mAP50 0.9628 vs float 0.9629) while 4-bit WEIGHTS collapse
under plain calibration (0.1973 overall, 0.0061 close). Activations are not the
bottleneck — W4A8 and W4A4 score alike — so this recovers 4-bit weights.

Why a standalone loop instead of yolov5/train.py
  train.py is structurally incompatible with a Brevitas model in three places:
    - it checkpoints with `torch.save({'model': deepcopy(model).half()})`;
      Brevitas quantizer classes are generated at runtime (brevitas.inject.*)
      and do not pickle, and .half() corrupts the quantizer state.
    - `strip_optimizer` at the end repeats that save.
    - it trains under torch.cuda.amp autocast; fake-quant scales in fp16 are
      unreliable.
  Everything that defines the training regime is still YOLOv5's: create_dataloader
  (same mosaic/scale/translate hyp), ComputeLoss, and val.run for scoring.

Deliberate omissions vs train.py, so the comparison stays honest
  - no AMP: fp32 throughout (see above).
  - no EMA: it deepcopies the model every step, which is both slow and fragile
    across Brevitas' dynamic classes. The float baseline used EMA, so expect a
    small handicap unrelated to quantization.
  - no warmup: we fine-tune from converged weights at a low lr, so the warmup
    that protects a cold-start run has nothing to do here.
"""
import argparse
import csv
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'yolov5'))
sys.path.insert(0, str(ROOT))

import val  # noqa: E402
from utils.dataloaders import create_dataloader  # noqa: E402
from utils.general import check_dataset, labels_to_class_weights, one_cycle  # noqa: E402
from utils.loss import ComputeLoss  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402
from brevitas.core.scaling import ParameterFromRuntimeStatsScaling  # noqa: E402

from qat.quantize import load_and_quantize  # noqa: E402
from qat.ptq_baseline import calibrate  # noqa: E402


def qat_optimizer(model, lr, momentum, decay):
    """YOLOv5's 3-group SGD, plus a 4th: Brevitas scales, never weight-decayed.

    smart_optimizer's `else` branch is a catch-all that would sweep the 58
    learned activation scales (ParameterFromRuntimeStatsScaling.value) into the
    decay group along with conv weights. Decaying a quantization scale drags it
    toward zero and collapses the activation range, so they get their own group.
    """
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    g_decay, g_nodecay, g_bias, g_scale = [], [], [], []
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if isinstance(v, ParameterFromRuntimeStatsScaling):
                g_scale.append(p)
            elif p_name == 'bias':
                g_bias.append(p)
            elif p_name == 'weight' and isinstance(v, bn):
                g_nodecay.append(p)
            else:
                g_decay.append(p)
    opt = torch.optim.SGD(g_bias, lr=lr, momentum=momentum, nesterov=True)
    opt.add_param_group({'params': g_decay, 'weight_decay': decay})
    opt.add_param_group({'params': g_nodecay, 'weight_decay': 0.0})
    opt.add_param_group({'params': g_scale, 'weight_decay': 0.0})
    print(f'optimizer: SGD(lr={lr}) groups -> {len(g_decay)} weight(decay={decay}), '
          f'{len(g_nodecay)} bn, {len(g_bias)} bias, {len(g_scale)} quant-scale(decay=0)')
    return opt


def fitness(mp, mr, map50, map95):
    """YOLOv5's checkpoint-selection metric (utils/metrics.py fitness)."""
    return 0.1 * map50 + 0.9 * map95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(ROOT / 'runs/train/relu_more_data_5/weights/best.pt'))
    ap.add_argument('--data', default=str(ROOT / 'configs/drone.yaml'))
    ap.add_argument('--hyp', default=str(ROOT / 'configs/hyp_gen_relu_more_data_5.yaml'))
    ap.add_argument('--weight-bits', type=int, default=4)
    ap.add_argument('--act-bits', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--lr0', type=float, default=0.003, help='10x below the float run; fine-tune, not cold start')
    ap.add_argument('--lrf', type=float, default=0.01)
    ap.add_argument('--calib-batches', type=int, default=16)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--device', default='0')
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--name', default='qat_w4a8')
    args = ap.parse_args()

    save_dir = ROOT / 'runs/qat' / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device, batch_size=args.batch_size)
    hyp = yaml.safe_load(Path(args.hyp).read_text())
    hyp['lr0'] = args.lr0
    hyp['lrf'] = args.lrf

    model, stats = load_and_quantize(args.weights, args.weight_bits, args.act_bits, device)
    print(f'quantized W{args.weight_bits}A{args.act_bits}: {stats}')

    data = check_dataset(args.data)
    nc = int(data['nc'])
    gs = max(int(model.stride.max()), 32)

    train_loader, dataset = create_dataloader(
        data['train'], args.imgsz, args.batch_size, gs, hyp=hyp, augment=True,
        workers=args.workers, shuffle=True, prefix='train: ')
    val_loader = create_dataloader(
        data['val'], args.imgsz, args.batch_size, gs, pad=0.5, rect=True,
        workers=args.workers, prefix='val: ')[0]

    # activation scales must be initialised from real data BEFORE training;
    # an uncalibrated quantizer starts from a meaningless range.
    print(f'calibrating on {args.calib_batches * args.batch_size} images ...')
    calibrate(model, train_loader, args.calib_batches, device)

    model.nc = nc
    model.hyp = hyp
    model.names = data['names']
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    compute_loss = ComputeLoss(model)

    optimizer = qat_optimizer(model, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    lf = one_cycle(1, hyp['lrf'], args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    csv_path = save_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'P', 'R', 'mAP50', 'mAP50-95', 'fitness', 'lr'])

    best_fit, best_epoch = -1.0, -1
    for epoch in range(args.epochs):
        model.train()
        t0, running, nb = time.time(), 0.0, len(train_loader)
        for i, (imgs, targets, _, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True).float() / 255
            loss, _ = compute_loss(model(imgs), targets.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            if i % 50 == 0:
                print(f'  ep{epoch} {i}/{nb} loss {running / (i + 1):.4f}', flush=True)
        scheduler.step()

        (mp, mr, map50, map95, *_), _, _ = val.run(
            data=data, model=model, dataloader=val_loader, imgsz=args.imgsz,
            batch_size=args.batch_size, device=device, half=False, plots=False,
            verbose=False, save_dir=Path(''))
        fit = fitness(mp, mr, map50, map95)
        lr_now = optimizer.param_groups[0]['lr']
        print(f'EPOCH {epoch}  loss {running / nb:.4f}  P {mp:.4f} R {mr:.4f} '
              f'mAP50 {map50:.4f} mAP50-95 {map95:.4f}  fitness {fit:.4f}  '
              f'lr {lr_now:.5f}  ({time.time() - t0:.0f}s)', flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{running / nb:.5f}', f'{mp:.5f}', f'{mr:.5f}',
                                    f'{map50:.5f}', f'{map95:.5f}', f'{fit:.5f}', f'{lr_now:.6f}'])

        if fit > best_fit:
            best_fit, best_epoch = fit, epoch
            torch.save({'state_dict': deepcopy(model).state_dict(), 'epoch': epoch,
                        'fitness': fit, 'metrics': [mp, mr, map50, map95],
                        'float_weights': args.weights, 'quant_stats': stats,
                        'weight_bits': args.weight_bits, 'act_bits': args.act_bits},
                       save_dir / 'best.pt')
        if epoch - best_epoch >= args.patience:
            print(f'early stop: no fitness gain in {args.patience} epochs')
            break

    print(f'\nBEST epoch {best_epoch}  fitness {best_fit:.4f} -> {save_dir / "best.pt"}')


if __name__ == '__main__':
    main()
