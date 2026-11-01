"""Opt-in guide supervision for Haruka's two salient posterior slots."""

from __future__ import annotations

from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NEIGHBOR_INDEX_KEY = "spatial_neighbor_idx"


def install_perturbation_supervision(module, labels_by_row, n_classes, weight=10.0):
    """Shape s and s_sp by observed guide class without decoder label injection.

    Graphs are isolated by slice x sgRNA, so every registered neighbor of a target
    cell has the same context label as its center.  The global neighbor index is
    used only to look up that label for the loss; it is never concatenated to an
    encoder or decoder input.  Leave-one-slice-out scoring remains the guard
    against spatial or slice memorization.
    """
    if n_classes < 2 or weight < 0:
        raise ValueError("need >=2 classes and a non-negative supervision weight")
    labels = torch.as_tensor(np.asarray(labels_by_row), dtype=torch.long)
    module.register_buffer("perturbation_label_by_row", labels, persistent=False)
    module.salient_perturbation_head = nn.Linear(module.n_salient_latent, n_classes)
    module.salient_spatial_perturbation_head = nn.Linear(module.n_spatial_latent, n_classes)
    original_loss = module.loss

    def supervised_loss(
        self, concat_tensors, inference_outputs, generative_outputs, kl_weight=1.0
    ):
        output = original_loss(
            concat_tensors, inference_outputs, generative_outputs, kl_weight
        )
        if not self.background_only_training and weight > 0:
            neighbor_index = concat_tensors["target"][NEIGHBOR_INDEX_KEY].long()
            label = self.perturbation_label_by_row[neighbor_index[:, 0]]
            valid = label >= 0
            if valid.any():
                salient = inference_outputs["target"]["qs_m"]
                salient = salient.reshape(-1, salient.shape[-1])
                logits_s = self.salient_perturbation_head(salient[valid])
                losses = [F.cross_entropy(logits_s, label[valid])]
                accuracies = [(logits_s.argmax(-1) == label[valid]).float().mean()]
                if self.spatial_attach == "both":
                    salient_spatial = inference_outputs["target"]["qssp_m"]
                    salient_spatial = salient_spatial.reshape(-1, salient_spatial.shape[-1])
                    logits_sp = self.salient_spatial_perturbation_head(
                        salient_spatial[valid]
                    )
                    losses.append(F.cross_entropy(logits_sp, label[valid]))
                    accuracies.append(
                        (logits_sp.argmax(-1) == label[valid]).float().mean()
                    )
                guide_loss = torch.stack(losses).mean()
                object.__setattr__(output, "loss", output.loss + weight * guide_loss)
                output.extra_metrics["perturbation_supervision_loss"] = guide_loss.detach()
                output.extra_metrics["perturbation_supervision_accuracy"] = (
                    torch.stack(accuracies).mean().detach()
                )
        return output

    module.loss = MethodType(supervised_loss, module)
    module.perturbation_supervision_weight = float(weight)
    module.n_perturbation_classes = int(n_classes)
    return module
