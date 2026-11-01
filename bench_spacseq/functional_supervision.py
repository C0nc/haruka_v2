"""Anchor-only target-function learning with within-function guide invariance."""

from __future__ import annotations

from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NEIGHBOR_INDEX_KEY = "spatial_neighbor_idx"


def _cosine_ce(latent, head, labels, scale=10.0):
    latent = F.normalize(latent, dim=-1)
    weight = F.normalize(head.weight, dim=-1)
    return F.cross_entropy(scale * F.linear(latent, weight), labels)


def _guide_moment_loss(latent, function, guide):
    """Align independent guides only inside their shared target function."""
    latent = F.normalize(latent, dim=-1)
    terms = []
    for function_id in torch.unique(function):
        in_function = function == function_id
        guide_ids = torch.unique(guide[in_function])
        guide_ids = guide_ids[guide_ids >= 0]
        if len(guide_ids) < 2:
            continue
        summaries = []
        for guide_id in guide_ids:
            group = latent[in_function & (guide == guide_id)]
            if len(group) >= 2:
                summaries.append((group.mean(0), group.var(0, unbiased=False)))
        for left in range(len(summaries)):
            for right in range(left + 1, len(summaries)):
                mean_left, var_left = summaries[left]
                mean_right, var_right = summaries[right]
                terms.append(
                    (mean_left - mean_right).square().mean()
                    + 0.25 * (var_left - var_right).square().mean()
                )
    if not terms:
        return latent.sum() * 0.0
    return torch.stack(terms).mean()


def install_functional_supervision(
    module,
    function_by_row,
    guide_by_row,
    n_functions,
    function_weight=5.0,
    guide_invariance_weight=2.0,
):
    """Pull independent guides toward a common function, never guide identity."""
    functions = torch.as_tensor(np.asarray(function_by_row), dtype=torch.long)
    guides = torch.as_tensor(np.asarray(guide_by_row), dtype=torch.long)
    module.register_buffer("function_label_by_row", functions, persistent=False)
    module.register_buffer("guide_identity_by_row", guides, persistent=False)
    module.salient_function_head = nn.Linear(
        module.n_salient_latent, n_functions, bias=False
    )
    module.salient_spatial_function_head = nn.Linear(
        module.n_spatial_latent, n_functions, bias=False
    )
    original_loss = module.loss

    def functional_loss(
        self, concat_tensors, inference_outputs, generative_outputs, kl_weight=1.0
    ):
        output = original_loss(
            concat_tensors, inference_outputs, generative_outputs, kl_weight
        )
        if self.background_only_training:
            return output
        center_row = concat_tensors["target"][NEIGHBOR_INDEX_KEY].long()[:, 0]
        function = self.function_label_by_row[center_row]
        guide = self.guide_identity_by_row[center_row]
        valid = (function >= 0) & (guide >= 0)
        if not valid.any():
            return output

        salient = inference_outputs["target"]["qs_m"]
        salient = salient.reshape(-1, salient.shape[-1])[valid]
        slots = [(salient, self.salient_function_head)]
        if self.spatial_attach == "both":
            salient_spatial = inference_outputs["target"]["qssp_m"]
            salient_spatial = salient_spatial.reshape(
                -1, salient_spatial.shape[-1]
            )[valid]
            slots.append((salient_spatial, self.salient_spatial_function_head))

        function_losses = [
            _cosine_ce(latent, head, function[valid]) for latent, head in slots
        ]
        guide_losses = [
            _guide_moment_loss(latent, function[valid], guide[valid])
            for latent, _ in slots
        ]
        function_loss = torch.stack(function_losses).mean()
        guide_loss = torch.stack(guide_losses).mean()
        object.__setattr__(
            output,
            "loss",
            output.loss
            + function_weight * function_loss
            + guide_invariance_weight * guide_loss,
        )
        output.extra_metrics["target_function_loss"] = function_loss.detach()
        output.extra_metrics["within_function_guide_invariance"] = guide_loss.detach()
        module_function_accuracy = []
        for latent, head in slots:
            logits = F.linear(F.normalize(latent, dim=-1), F.normalize(head.weight, dim=-1))
            module_function_accuracy.append(
                (logits.argmax(-1) == function[valid]).float().mean()
            )
        output.extra_metrics["target_function_accuracy"] = (
            torch.stack(module_function_accuracy).mean().detach()
        )
        return output

    module.loss = MethodType(functional_loss, module)
    module.function_supervision_weight = float(function_weight)
    module.guide_invariance_weight = float(guide_invariance_weight)
    module.n_target_functions = int(n_functions)
    return module
