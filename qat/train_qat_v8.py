"""Brevitas QAT fine-tuning for YOLOv8n-P3 (ReLU6), through Ultralytics' trainer.

Target is W4A4 (body 4-bit, stem and Detect head 8-bit) — the reference recipe.
`qat/ptq_baseline`-style calibration is not needed here: the activation scales are
constants and the weight scales come from the weights, so plain PTQ is just
"quantize and run". It gives W8A8 for free but costs W4A4 3.9 pt of close
mAP50-95 and 10.3 pt of long-range mAP50. This recovers that.

Why a trainer subclass, where the YOLOv5 version needed a standalone loop
  `qat/train_qat.py` reimplements the loop because yolov5's train.py fights a
  Brevitas model in three places. Two of the three are gone here:
    - optimizer groups: our quantizer scales are CONST, so there are zero
      learnable quantization parameters to keep out of weight decay.
    - AMP: still a problem, so it is switched off (`amp=False`).
    - checkpointing: still a problem, and `save_model` below is the fix.
  Everything else — mosaic, the DFL loss, the scheduler, EMA, the dataloaders,
  the per-epoch validation, the W&B callback — is reused as-is, which keeps the
  regime identical to the float run it is fine-tuning from.

  `BaseTrainer.setup_model` returns immediately when `self.model` is already an
  nn.Module, so the quantized net is simply assigned before `train()`; no
  `get_model` override is needed.

THE CHECKPOINT RULES (both learned the hard way)
  1. Ultralytics saves `deepcopy(unwrap_model(ema.ema)).half()`. Brevitas builds
     its quantizer classes at runtime under `brevitas.inject`, so the pickle is
     fragile, and .half() is not something to do to quantizer state. `save_model`
     writes a plain fp32 state_dict plus the bit widths instead — same shape as
     the yolov5 QAT checkpoints, so `scripts/center_error.py` can grow one branch
     to read both.
  2. RESTORE ON CPU, THEN MOVE. Brevitas' const-scale buffers are rebuilt on CPU
     by `load_state_dict`, so loading into a CUDA model strands 120 of them there
     and the next forward dies with "expected all tensors on the same device".
     Build on CPU, `load_state_dict`, and only then `.to(device)`. Verified
     bit-exact that way; see `load_qat_checkpoint`.
"""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionTrainer

from qat.quantize_v8 import load_and_quantize

ROOT = Path(__file__).resolve().parents[1]


class QATDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that checkpoints a Brevitas model safely."""

    qat_meta = {}

    def save_model(self):
        ckpt = {
            'state_dict': deepcopy(self.ema.ema).float().state_dict(),
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'train_metrics': {**self.metrics, 'fitness': self.fitness},
            'train_args': vars(self.args),
            **self.qat_meta,
        }
        self.wdir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)

    def final_eval(self):
        """Validate the in-memory EMA instead of reloading best.pt.

        The stock version calls `strip_optimizer` and then hands the FILE to the
        validator, which reloads it through `load_checkpoint` and wants a pickled
        `model`/`ema` entry. Our checkpoints carry a state_dict (see the module
        docstring), so that path raises KeyError: 'model'. The EMA weights are
        already the ones that were saved, so validate those directly.
        """
        self.validator.args.plots = self.args.plots
        self.metrics = self.validator(model=deepcopy(self.ema.ema).float().eval())
        self.metrics.pop('fitness', None)
        self.run_callbacks('on_fit_epoch_end')


def load_qat_checkpoint(path, device='cpu'):
    """Rebuild a quantized model from a `save_model` checkpoint.

    Builds and loads on CPU before moving — see the module docstring, rule 2.
    """
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model, _ = load_and_quantize(ck['float_weights'], ck['low_bits'], ck['high_bits'], 'cpu')
    sd = ck['state_dict']
    # Checkpoints written before the input quantizer was dropped carry extra
    # `input_quant.*` buffers. They were constants with no learnable parameters
    # and no effect on the weights, so the rest of the checkpoint stays valid.
    own = set(model.state_dict())
    extra = [k for k in sd if k not in own]
    assert all('.input_quant.' in k for k in extra), f'unexpected extra keys: {extra[:5]}'
    model.load_state_dict({k: v for k, v in sd.items() if k in own}, strict=True)
    return model.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='runs/train/v8n_p3_relu62/weights/best.pt',
                    help='float ReLU6 checkpoint to fine-tune from')
    ap.add_argument('--data', default='configs/drone.yaml')
    ap.add_argument('--low-bits', type=int, default=4, help='network body')
    ap.add_argument('--high-bits', type=int, default=8, help='stem + Detect head')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--batch', type=int, default=192)
    ap.add_argument('--lr0', type=float, default=0.002,
                    help='fine-tuning from converged weights, so well below the float run')
    ap.add_argument('--mosaic', type=float, default=0.5)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--name', default=None)
    ap.add_argument('--project', default=str(ROOT / 'runs' / 'qat'))
    ap.add_argument('--wandb-project', default='YOLOv8')
    ap.add_argument('--no-wandb', action='store_true')
    args = ap.parse_args()

    name = args.name or f'v8n_p3_w{args.low_bits}a{args.low_bits}'

    # Open the W&B run ourselves. Ultralytics' callback is guarded by
    # `if not wb.run:` and would otherwise name the project after `args.project`
    # — which must be an absolute path here (build_notes §10.9).
    if not args.no_wandb:
        try:
            import wandb
            from ultralytics.utils import SETTINGS
            if not SETTINGS.get('wandb'):
                SETTINGS.update({'wandb': True})
            wandb.init(project=args.wandb_project, name=name, config=vars(args))
        except Exception as e:                        # never let logging kill a run
            print('wandb init skipped:', e)

    # Build on CPU; the trainer moves it to the device itself.
    model, stats = load_and_quantize(args.weights, args.low_bits, args.high_bits, 'cpu')
    print(f'quantized W{args.low_bits}A{args.low_bits} '
          f'(stem/head W{args.high_bits}A{args.high_bits}): {stats}')

    trainer = QATDetectionTrainer(overrides=dict(
        model=args.weights, data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers,
        project=args.project, name=name, exist_ok=True,
        optimizer='SGD', lr0=args.lr0, mosaic=args.mosaic,
        warmup_epochs=0.0,      # fine-tuning converged weights, nothing to warm up
        amp=False,              # fake-quant scales under fp16 autocast are unreliable
        val=True, plots=False,
    ))
    if args.no_wandb:
        # Ultralytics' W&B callback keys off the global SETTINGS, so skipping our
        # own wandb.init() is not enough — it would open its own run and name the
        # project after the output path. Drop the callbacks instead of flipping
        # SETTINGS, which would persist to ~/.config/Ultralytics/settings.json.
        for event, fns in trainer.callbacks.items():
            trainer.callbacks[event] = [f for f in fns if 'callbacks.wb' not in f.__module__]

    trainer.qat_meta = {'float_weights': str(Path(args.weights).resolve()),
                        'low_bits': args.low_bits, 'high_bits': args.high_bits,
                        'quantize_stats': stats}
    trainer.model = model
    trainer.train()

    print('\nrun dir:', trainer.save_dir)
    print('best   :', trainer.best)


if __name__ == '__main__':
    main()
