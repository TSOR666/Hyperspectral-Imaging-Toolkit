"""Opt-in diagnostics for actual optimizer movement and residual contributions."""
from __future__ import annotations

import math
from typing import Dict, Iterable

import torch
from torch import nn


class TrainingHealthMonitor:
    """Sample a step without changing model computation or retaining its graph.

    A snapshot uses one extra copy of the trainable weights. Call only on
    diagnostic steps, after backward and immediately before optimizer.step().
    Ratios are group L2 norms, not averages of per-tensor ratios. Zero-initialized
    gates are listed separately so their large relative updates are interpretable.
    """

    def __init__(self, model: nn.Module, groups: Iterable[str] | None = None):
        self.model = model
        self.groups = tuple(groups or self._default_groups())
        self.activations: Dict[str, dict] = {}
        self.enabled = True
        self._before: Dict[str, torch.Tensor] = {}
        self._handles = []
        modules = dict(model.named_modules())
        for name in self.groups:
            if name in modules:
                self._handles.append(modules[name].register_forward_hook(self._hook(name)))
        for name, module in model.named_modules():
            if callable(getattr(module, "attention_operator", None)):
                self._handles.append(module.register_forward_hook(self._operator_hook(name)))

    def _default_groups(self):
        yield ""  # whole model
        for name, module in self.model.named_modules():
            if name and (
                name in {"encoder1", "encoder2", "bottleneck", "decoder1", "decoder2",
                         "spectral_output", "refine_stage", "output_proj", "to_spectral", "input_skip", "stem", "head", "output_head"}
                or (name.startswith(("encoder_stages.", "decoder_stages.")) and name.count(".") == 1)
                or (name.startswith(("encoder_stages.", "stages.")) and name.count(".") == 2)
                or (name.startswith("fusion.") and name.count(".") == 1)
            ):
                yield name

    @staticmethod
    def _rms(value: torch.Tensor) -> float:
        return float(value.detach().float().square().mean().sqrt().item())

    def _hook(self, name):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return
            if not torch.isfinite(output).all():
                raise FloatingPointError(f"Non-finite activation in {name or 'model'}")
            record = {"output_rms": self._rms(output)}
            if inputs and isinstance(inputs[0], torch.Tensor):
                value = inputs[0]
                record["input_rms"] = self._rms(value)
                if value.shape == output.shape:
                    record["residual_ratio"] = self._rms(output - value) / (record["input_rms"] + 1e-12)
            self.activations[name or "model"] = record
        return hook

    def _operator_hook(self, name):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            height, width = inputs[0].shape[-2:]
            self.activations[name] = {
                "height": height, "width": width,
                "attention_operator": module.attention_operator(height, width),
            }
        return hook

    @torch.no_grad()
    def before_step(self):
        self._before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }

    @torch.no_grad()
    def after_step(self) -> dict:
        if not self._before:
            raise RuntimeError("Call before_step immediately before optimizer.step")
        stats = {name: [0.0, 0.0, 0.0] for name in self.groups}
        gates = {}
        for name, parameter in self.model.named_parameters():
            before = self._before.get(name)
            if before is None:
                continue
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"Non-finite parameter after step: {name}")
            movement = float((parameter.float() - before.float()).square().sum().item())
            weight = float(before.float().square().sum().item())
            grad = float(parameter.grad.float().square().sum().item()) if parameter.grad is not None else 0.0
            for group, values in stats.items():
                if not group or name.startswith(group + "."):
                    values[0] += movement
                    values[1] += weight
                    values[2] += grad
            if name.endswith((".gate", ".gamma", ".gamma2", ".ls1", ".ls2", ".ls3")):
                gates[name] = {"mean": float(parameter.mean().item()),
                               "max_abs": float(parameter.abs().max().item())}
        self._before.clear()
        return {
            "groups": {
                group or "model": {
                    "update_ratio": math.sqrt(values[0]) / (math.sqrt(values[1]) + 1e-12),
                    "gradient_norm": math.sqrt(values[2]),
                } for group, values in stats.items()
            },
            "activations": dict(self.activations), "gates": gates,
        }

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._before.clear()
