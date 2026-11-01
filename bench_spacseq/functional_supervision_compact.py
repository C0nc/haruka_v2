"""Compact anchor-only target-function learning with guide invariance."""

from __future__ import annotations

from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NEIGHBOR_INDEX_KEY = "spatial_neighbor_idx"


def cosine_ce(latent, head, labels, scale=10.0):
    latent = F.normalize(latent, dim=-1)
    weight = F.normalize(head.weight, dim=-1)
    return F.cross_entropy(scale * F.linear(latent, weight), labels)


def prototype_compactness(latent, head, labels):
    latent = F.normalize(latent, dim=-1)
    prototype = F.normalize(head.weight, dim=-1)[labels]
    return (1.0 - (latent * prototype).sum(-1)).mean()


def guide_moment_loss(latent, function, guide):
    latent = F.normalize(latent, dim=-1)
    terms = []
    for function_id in torch.unique(function):
        in_function = function == function_id
        summaries = []
        for guide_id in torch.unique(guide[in_function]):
            group = latent[in_function & (guide == guide_id)]
            if guide_id >= 0 and len(group) >= 2:
                summaries.append((group.mean(0), group.var(0, unbiased=False)))
        for left in range(len(summaries)):
            for right in range(left + 1, len(summaries)):
                lm, lv = summaries[left]
                rm, rv = summaries[right]
                terms.append(
                    (lm - rm).square().mean()
                    + 0.25 * (lv - rv).square().mean()
                )
    return torch.stack(terms).mean() if terms else latent.sum() * 0.0


def install_functional_supervision(
    module,
    function_by_row,
    guide_by_row,
    n_functions,
    function_weight=5.0,
    guide_invariance_weight=2.0,
    compactness_weight=5.0,
):
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
        center = concat_tensors["target"][NEIGHBOR_INDEX_KEY].long()[:, 0]
        function = self.function_label_by_row[center]
        guide = self.guide_identity_by_row[center]
        valid = (function >= 0) & (guide >= 0)
        if not valid.any():
            return output

        salient = inference_outputs["target"]["qs_m"]
        salient = salient.reshape(-1, salient.shape[-1])[valid]
        slots = [(salient, self.salient_function_head)]
        if self.spatial_attach == "both":
            spatial = inference_outputs["target"]["qssp_m"]
            spatial = spatial.reshape(-1, spatial.shape[-1])[valid]
            slots.append((spatial, self.salient_spatial_function_head))

        ce = [cosine_ce(z, head, function[valid]) for z, head in slots]
        compact = [
            prototype_compactness(z, head, function[valid]) for z, head in slots
        ]
        invariance = [
            guide_moment_loss(z, function[valid], guide[valid]) for z, _ in slots
        ]
        ce_loss = torch.stack(ce).mean()
        # Double weight on intrinsic compactness; it carries more cell-state
        # variation and otherwise learns boundaries rather than clusters.
        compact_loss = (2.0 * compact[0] + sum(compact[1:])) / (1 + len(compact))
        invariance_loss = torch.stack(invariance).mean()
        object.__setattr__(
            output,
            "loss",
            output.loss
            + function_weight * ce_loss
            + compactness_weight * compact_loss
            + guide_invariance_weight * invariance_loss,
        )
        output.extra_metrics["target_function_loss"] = ce_loss.detach()
        output.extra_metrics["target_function_compactness"] = compact_loss.detach()
        output.extra_metrics["within_function_guide_invariance"] = invariance_loss.detach()
        return output

    module.loss = MethodType(functional_loss, module)
    module.function_supervision_weight = float(function_weight)
    module.function_compactness_weight = float(compactness_weight)
    module.guide_invariance_weight = float(guide_invariance_weight)
    module.n_target_functions = int(n_functions)
    return module
