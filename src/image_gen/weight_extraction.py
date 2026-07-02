"""Deterministic extraction of trainable parameters from state dictionaries."""

from __future__ import annotations

import re
from typing import Any

import torch


def _natural_key(text: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def _parameter_sort_key(name: str) -> tuple[Any, int]:
    base, kind = name.rsplit(".", 1)
    kind_order = {"weight": 0, "bias": 1}.get(kind, 2)
    return _natural_key(base), kind_order


def unwrap_state_dict(checkpoint_or_state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return the raw state_dict from either a checkpoint or a state_dict."""
    if "state_dict" in checkpoint_or_state_dict:
        return checkpoint_or_state_dict["state_dict"]
    return checkpoint_or_state_dict


def extract_parameters_from_state_dict(
    checkpoint_or_state_dict: dict[str, Any],
    include_bias: bool = True,
) -> list[dict[str, Any]]:
    """Extract ordered weights and optional biases as CPU float tensors."""
    state_dict = unwrap_state_dict(checkpoint_or_state_dict)
    names = [
        name
        for name, tensor in state_dict.items()
        if isinstance(tensor, torch.Tensor)
        and (name.endswith(".weight") or name.endswith(".bias"))
        and (include_bias or not name.endswith(".bias"))
    ]
    names = sorted(names, key=_parameter_sort_key)

    parameters: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        tensor = state_dict[name].detach().cpu().float().clone()
        kind = name.rsplit(".", 1)[-1]
        parameters.append(
            {
                "index": index,
                "name": name,
                "kind": kind,
                "tensor": tensor,
                "shape": list(tensor.shape),
            }
        )
    return parameters

