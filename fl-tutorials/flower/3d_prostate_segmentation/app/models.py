# Copyright (c) 2026 Guy's and St Thomas' NHS Foundation Trust & King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""The segmentation network, built from the nnU-Net plan that ships beside this file.

nnU-Net's *experiment planner* (``calculate_dataset_fingerprint_segmentation.py``, run once, offline,
over every participating site) decides the U-Net topology — how many resolution stages, which kernel
sizes and pooling strides at each, how many feature maps — from the cohort's voxel spacing and
shape. Its output, ``nnUNetPlans_segmentation.json``, is committed next to this module, and this is
the **one** place that turns it into a ``torch.nn.Module``: the ServerApp calls ``get_model()`` to
create the initial global weights and every ClientApp calls the same zero-argument factory before
loading the weights it receives, so the two can never disagree about the architecture.

The network class is MONAI's ``DynUNet`` — the nnU-Net topology, re-implemented in MONAI — rather
than nnU-Net's own ``PlainConvUNet``: the FL images carry MONAI but not ``nnunetv2``, and an app
must not install packages at run time. The mapping from the plan's ``arch_kwargs`` to ``DynUNet``'s
constructor is spelled out in ``build_dynunet_from_plan``; anything the plan asks for that DynUNet
cannot express raises rather than being silently approximated, because a topology mismatch between
sites is exactly what makes weight aggregation impossible (see the README's "nnU-Net plans").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import DynUNet
from torch import nn

# The plan the planner wrote; committed so server and clients build the same net.
PLAN_PATH = Path(__file__).with_name("nnUNetPlans_segmentation.json")
# One t2w channel in; three overlapping label channels out — [whole gland, PZ, TZ], the order
# `dataset.PicaiDataset.combine_masks` stacks them in.
IN_CHANNELS = 1
OUT_CHANNELS = 3
CONFIGURATION = "3d_fullres"


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    """Read the nnU-Net plan JSON (``nnUNetPlans_segmentation.json``).

    Args:
        path: The plan file. Defaults to the one committed beside this module.

    Returns:
        The parsed plan.

    Raises:
        FileNotFoundError: Naming the planner command to run, when the plan has not been generated.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"nnU-Net plan not found at {path}. Generate it with `make plan` in the tutorial directory "
            "(runs calculate_dataset_fingerprint_segmentation.py over every site) and commit the result."
        )
    with open(path) as handle:
        return json.load(handle)


def _configuration(plan: dict[str, Any], name: str = CONFIGURATION) -> dict[str, Any]:
    try:
        return plan["configurations"][name]
    except KeyError as err:
        raise KeyError(f"plan has no configuration {name!r}: {sorted(plan.get('configurations', {}))}") from err


def build_dynunet_from_plan(
    plan: dict[str, Any],
    *,
    in_channels: int = IN_CHANNELS,
    out_channels: int = OUT_CHANNELS,
    deep_supervision: bool = True,
) -> DynUNet:
    """Instantiate a ``DynUNet`` with the topology the plan's ``3d_fullres`` architecture describes.

    The plan's ``kernel_sizes`` / ``strides`` are per stage in nnU-Net's transposed ``(z, y, x)``
    order; the training loop feeds the network ``(B, C, z, y, x)`` tensors (``train_helpers.train_seg``
    rearranges ``b c h w d -> b c d h w``), so they are passed through unchanged.

    Mapping (plan → DynUNet):

    * ``kernel_sizes`` → ``kernel_size``; ``strides`` → ``strides``; ``strides[1:]`` →
      ``upsample_kernel_size`` (a transposed conv undoes each pooling step).
    * ``features_per_stage`` → ``filters``.
    * ``norm_op`` (``InstanceNorm3d`` + its kwargs) → ``norm_name=("instance", {...})``; ``nonlin``
      (``LeakyReLU``) → ``act_name=("leakyrelu", {...})``.
    * ``n_stages - 2`` → ``deep_supr_num``: nnU-Net supervises every decoder resolution but the
      lowest, i.e. ``n_stages - 1`` outputs; DynUNet returns ``1 + deep_supr_num``.

    Args:
        plan: A parsed plan (``load_plan``).
        in_channels: Input channels (one t2w volume).
        out_channels: Output channels (whole gland, PZ, TZ).
        deep_supervision: Whether the net returns the auxiliary decoder outputs in training mode.
            Toggle later with ``set_deep_supervision_enabled``.

    Returns:
        The network, on the CPU, randomly initialised.

    Raises:
        ValueError: When the plan asks for something DynUNet cannot build faithfully — a residual
            encoder, more or fewer than two convolutions per stage, a non-3-D convolution, a norm
            or activation other than instance norm / leaky ReLU, or fewer than three stages.
    """
    architecture = _configuration(plan)["architecture"]
    arch = architecture["arch_kwargs"]

    if "n_blocks_per_stage" in arch:
        raise ValueError(
            f"plan architecture {architecture.get('network_class_name')} is a residual-encoder U-Net; "
            "DynUNet's basic blocks cannot reproduce it — plan with the default PlainConvUNet."
        )
    n_stages = int(arch["n_stages"])
    if n_stages < 3:
        raise ValueError(f"plan has {n_stages} stage(s); deep supervision needs at least 3.")
    per_stage = list(arch["n_conv_per_stage"]) + list(arch["n_conv_per_stage_decoder"])
    if any(int(n) != 2 for n in per_stage):
        raise ValueError(f"plan uses {per_stage} convolutions per stage; DynUNet's UnetBasicBlock is fixed at two.")
    if not str(arch["conv_op"]).endswith("Conv3d"):
        raise ValueError(f"plan conv_op {arch['conv_op']!r} is not a 3-D convolution.")
    if not str(arch["norm_op"]).endswith("InstanceNorm3d"):
        raise ValueError(f"plan norm_op {arch['norm_op']!r} is not InstanceNorm3d.")
    if str(arch["nonlin"]) != "torch.nn.LeakyReLU":
        raise ValueError(f"plan nonlin {arch['nonlin']!r} is not torch.nn.LeakyReLU.")

    kernel_sizes = [list(k) for k in arch["kernel_sizes"]]
    strides = [list(s) for s in arch["strides"]]
    filters = [int(f) for f in arch["features_per_stage"]]
    if not len(kernel_sizes) == len(strides) == len(filters) == n_stages:
        raise ValueError(
            f"plan is inconsistent: n_stages={n_stages}, {len(kernel_sizes)} kernel sizes, "
            f"{len(strides)} strides, {len(filters)} feature widths."
        )

    norm_kwargs = dict(arch.get("norm_op_kwargs") or {})
    nonlin_kwargs = {"inplace": True, "negative_slope": 0.01, **dict(arch.get("nonlin_kwargs") or {})}

    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_sizes,
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=filters,
        norm_name=("instance", norm_kwargs),
        act_name=("leakyrelu", nonlin_kwargs),
        deep_supervision=deep_supervision,
        deep_supr_num=n_stages - 2,
        res_block=False,
    )


def get_model() -> nn.Module:
    """The zero-argument factory the ServerApp and every ClientApp share (built from ``PLAN_PATH``)."""
    return build_dynunet_from_plan(load_plan())


def set_deep_supervision_enabled(enabled: bool, is_ddp: bool = False, network: nn.Module | None = None) -> nn.Module:
    """Turn the auxiliary decoder outputs on or off — the same call shape ``network.py`` had.

    DynUNet reads its ``deep_supervision`` flag on every forward and keeps the auxiliary heads in
    ``state_dict`` either way, so toggling never changes the weights on the wire: a strict
    ``load_state_dict`` on the server and the clients stays valid whatever each side sets here.

    Args:
        enabled: True to return the auxiliary outputs in training mode.
        is_ddp: Unwrap a ``DistributedDataParallel`` module first. Kept for interface parity.
        network: The network to configure.

    Returns:
        The (unwrapped) network.
    """
    if network is None:
        raise ValueError("network is required")
    module = network.module if is_ddp else network
    module.deep_supervision = enabled
    return module


def split_deep_supervision_outputs(
    logits: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> list[torch.Tensor]:
    """Normalise a network output to nnU-Net's list-of-outputs convention, full resolution first.

    ``train_helpers.train_seg`` was written against nnU-Net, whose network returns a *list* of
    outputs when deep supervision is on. DynUNet returns them stacked along dim 1 instead — a
    ``(B, 1 + deep_supr_num, C, *spatial)`` tensor whose every head is already interpolated to full
    resolution — so this is the whole adapter: unbind it. Head 0 is the network's real output.

    Args:
        logits: Whatever the network returned.

    Returns:
        A list with the full-resolution output first, then the auxiliary heads (if any).
    """
    if isinstance(logits, (list, tuple)):
        return list(logits)
    if logits.ndim == 6:
        return list(logits.unbind(1))
    return [logits]


def deep_supervision_weights(num_outputs: int) -> list[float]:
    """nnU-Net's loss weights per output — ``1/2**i``, normalised to sum to one."""
    weights = [1.0 / (2**i) for i in range(num_outputs)]
    total = sum(weights)
    return [w / total for w in weights]
