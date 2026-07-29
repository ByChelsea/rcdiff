import math
import torch
import contextlib
from copy import deepcopy

class ModelEMA:
    """
    Maintains an exponential moving average (EMA) of model parameters.
    """
    def __init__(self, model, decay=0.9999, use_fp32=True, device=None,
                 update_every=1, warmup_steps=0):
        self.decay = float(decay)
        self.use_fp32 = bool(use_fp32)
        self.device = device
        self.update_every = int(update_every)
        self.warmup_steps = int(warmup_steps)
        self.num_updates = 0

        self._param_names = []
        self.shadow = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._param_names.append(n)
            t = p.detach().float().clone() if self.use_fp32 else p.detach().clone()
            if self.device is not None:
                t = t.to(self.device, non_blocking=True)
            self.shadow[n] = t

    def _current_decay(self):
        if self.warmup_steps <= 0 or self.num_updates >= self.warmup_steps:
            return self.decay
        warm = (self.num_updates + 1) / float(self.warmup_steps)
        return (1 - warm) * 0.0 + warm * self.decay

    @torch.no_grad()
    def maybe_update(self, model, global_step: int):
        if self.update_every <= 1 or (global_step + 1) % self.update_every == 0:
            self.update(model)

    @torch.no_grad()
    def update(self, model):
        d = self._current_decay()
        for n, p in model.named_parameters():
            if n not in self.shadow or (not p.requires_grad):
                continue
            src = p.detach().float() if self.use_fp32 else p.detach()
            if self.device is not None:
                src = src.to(self.device, non_blocking=True)
            self.shadow[n].mul_(d).add_(src, alpha=(1.0 - d))
        self.num_updates += 1

    @contextlib.contextmanager
    def apply_to(self, model):
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].to(p.dtype).to(p.device, non_blocking=True))

        try:
            yield
        finally:
            for n, p in model.named_parameters():
                if n in backup:
                    p.data.copy_(backup[n])

    def state_dict(self):
        return dict(
            decay=self.decay,
            use_fp32=self.use_fp32,
            device=str(self.device) if self.device is not None else None,
            update_every=self.update_every,
            warmup_steps=self.warmup_steps,
            num_updates=self.num_updates,
            shadow={k: v.cpu() for k, v in self.shadow.items()},
            param_names=self._param_names,
        )

    def load_state_dict(self, state):
        self.decay = float(state['decay'])
        self.use_fp32 = bool(state['use_fp32'])
        self.update_every = int(state.get('update_every', 1))
        self.warmup_steps = int(state.get('warmup_steps', 0))
        self.num_updates = int(state.get('num_updates', 0))
        self._param_names = list(state.get('param_names', []))

        device = self.device
        self.shadow = {k: v.to(device) if device else v for k, v in state['shadow'].items()}
