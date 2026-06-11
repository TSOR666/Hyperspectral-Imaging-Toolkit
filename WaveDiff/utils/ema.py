"""Exponential moving average support for model evaluation and checkpoints."""

from copy import deepcopy

import torch


class ModelEMA:
    """Maintain a detached exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.num_updates = 0
        self.module = deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            source_value = source[name].detach()
            if value.is_floating_point():
                value.lerp_(source_value, 1.0 - self.decay)
            else:
                value.copy_(source_value)

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "model": self.module.state_dict(),
        }

    def load_state_dict(self, state):
        if "model" in state:
            self.decay = float(state.get("decay", self.decay))
            self.num_updates = int(state.get("num_updates", 0))
            state = state["model"]
        self.module.load_state_dict(state)
