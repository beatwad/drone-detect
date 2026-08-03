"""Calibrate the quantized model on real data and measure the QAT starting point.

This is the "epoch 0" number for phase 4: INT8 weights + activations, activation
scales calibrated on training images, but NO fine-tuning yet. It tells us how much
QAT actually has to recover.

Calibration matters and cannot be skipped: Brevitas activation quantizers start
with an uninitialised scale and collect range statistics during a `calibration_mode`
forward pass. Evaluating before that measures nothing about the quantization.

`val.run(model=...)` is used rather than the `--weights` path because
`attempt_load` calls `.fuse()`, which folds BN into conv via `fuse_conv_and_bn` and
would replace our QuantConv2d with a plain nn.Conv2d. Passing the model object is
what train.py does and skips fusing entirely.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'yolov5'))
sys.path.insert(0, str(ROOT))

import val  # noqa: E402  (vendored yolov5)
from utils.dataloaders import create_dataloader  # noqa: E402
from utils.general import check_dataset  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402
from brevitas.graph.calibrate import calibration_mode  # noqa: E402

from qat.quantize import load_and_quantize  # noqa: E402


def calibrate(model, loader, n_batches, device):
    """Collect activation ranges over `n_batches` of real images."""
    model.eval()
    with torch.no_grad(), calibration_mode(model):
        for i, (im, _, _, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(im.to(device, non_blocking=True).float() / 255)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(ROOT / 'runs/train/relu_more_data_5/weights/best.pt'))
    ap.add_argument('--data', default=str(ROOT / 'configs/drone.yaml'))
    ap.add_argument('--weight-bits', type=int, default=8)
    ap.add_argument('--act-bits', type=int, default=8)
    ap.add_argument('--calib-batches', type=int, default=16)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0')
    ap.add_argument('--save', default=str(ROOT / 'runs/qat/calibrated.pt'))
    ap.add_argument('--regimes', default='overall,close,mid,long',
                    help='comma-separated subset to evaluate')
    args = ap.parse_args()

    device = select_device(args.device, batch_size=args.batch_size)
    model, stats = load_and_quantize(args.weights, args.weight_bits, args.act_bits, device)
    print(f'quantized: {stats}')

    data = yaml.safe_load(Path(args.data).read_text())
    gs = max(int(model.stride.max()), 32)

    calib_loader = create_dataloader(data['train'], args.imgsz, args.batch_size, gs,
                                     pad=0.5, rect=True, workers=8, prefix='calib: ')[0]
    n_imgs = args.calib_batches * args.batch_size
    print(f'calibrating on {n_imgs} training images ...')
    calibrate(model, calib_loader, args.calib_batches, device)

    # state_dict, not the model object: Brevitas builds its quantizer classes
    # dynamically (brevitas.inject.*), so pickling the module tree fails. Rebuild
    # with load_and_quantize(...) at the same bit widths, then load_state_dict.
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'quant_stats': stats,
                'float_weights': args.weights, 'calib_images': n_imgs,
                'weight_bits': args.weight_bits, 'act_bits': args.act_bits}, args.save)
    print(f'saved calibrated state_dict -> {args.save}')

    # val.run(model=...) takes train.py's code path: it expects `data` already
    # parsed to a dict and requires an explicit dataloader, because its own
    # loader-creation block only runs when called with --weights.
    want = args.regimes.split(',')
    for tag, cfg in (('overall', 'configs/drone.yaml'),
                     ('close', 'configs/drone_val_close.yaml'),
                     ('mid', 'configs/drone_val_mid.yaml'),
                     ('long', 'configs/drone_val_long.yaml')):
        if tag not in want:
            continue
        d = check_dataset(str(ROOT / cfg))
        loader = create_dataloader(d['val'], args.imgsz, args.batch_size, gs,
                                   pad=0.5, rect=True, workers=8, prefix=f'{tag}: ')[0]
        (mp, mr, map50, map95, *_), _, _ = val.run(
            data=d, model=model, dataloader=loader, imgsz=args.imgsz,
            batch_size=args.batch_size, device=device, half=False, plots=False,
            verbose=False, save_dir=Path(''))
        print(f'RESULT {tag:8} P {mp:.4f}  R {mr:.4f}  mAP50 {map50:.4f}  mAP50-95 {map95:.4f}')


if __name__ == '__main__':
    main()
