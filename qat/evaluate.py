"""Per-regime evaluation of a saved QAT checkpoint.

QAT checkpoints hold a state_dict, not a pickled model (Brevitas quantizer classes
are generated at runtime and do not pickle), so the graph is rebuilt at the same
bit widths and the weights loaded into it.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'yolov5'))
sys.path.insert(0, str(ROOT))

import val  # noqa: E402
from utils.dataloaders import create_dataloader  # noqa: E402
from utils.general import check_dataset  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

from qat.quantize import load_and_quantize  # noqa: E402

REGIMES = (('overall', 'configs/drone.yaml'),
           ('close', 'configs/drone_val_close.yaml'),
           ('mid', 'configs/drone_val_mid.yaml'),
           ('long', 'configs/drone_val_long.yaml'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(ROOT / 'runs/qat/qat_w4a8/best.pt'))
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    device = select_device(args.device, batch_size=args.batch_size)
    # Build and load on CPU, then move. Brevitas quantizers cache scale tensors as
    # plain attributes rather than buffers, so an nn.Module.to() issued BEFORE
    # load_state_dict leaves those caches on CPU and the first conv dies with a
    # cuda/cpu mismatch inside quant_output_scale_impl.
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model, _ = load_and_quantize(ck['float_weights'], ck['weight_bits'], ck['act_bits'], 'cpu')
    missing, unexpected = model.load_state_dict(ck['state_dict'], strict=False)
    assert not unexpected, f'unexpected keys: {unexpected[:5]}'
    model = model.to(device)
    print(f"W{ck['weight_bits']}A{ck['act_bits']} from {args.ckpt} "
          f"(epoch {ck.get('epoch')}, fitness {ck.get('fitness', float('nan')):.4f}); "
          f'missing keys: {len(missing)}')

    gs = max(int(model.stride.max()), 32)
    for tag, cfg in REGIMES:
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
