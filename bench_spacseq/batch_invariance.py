"""Opt-in four-slot batch invariance for Haruka.

Haruka already injects the registered batch one-hot into both intrinsic
encoders, the neighbor encoding path, and the decoder.  This module adds the
complementary requirement that posterior means do not retain a predictable
slice distribution.  It is installed on a model instance so legacy benchmark
configurations remain byte-for-byte unchanged.
"""

from __future__ import annotations

from types import MethodType

import torch
from scvi import REGISTRY_KEYS


def _four_slot_batch_mmd(module, inference_outputs, concat_tensors):
    bg, tg = inference_outputs["background"], inference_outputs["target"]
    bg_batch = concat_tensors["background"][REGISTRY_KEYS.BATCH_KEY].reshape(-1)
    tg_batch = concat_tensors["target"][REGISTRY_KEYS.BATCH_KEY].reshape(-1)

    def flat(value):
        return value.reshape(-1, value.shape[-1])

    def across_batches(value, batch):
        levels = torch.unique(batch)
        terms = []
        for left in range(levels.numel()):
            for right in range(left + 1, levels.numel()):
                x = value[batch == levels[left]]
                y = value[batch == levels[right]]
                if x.shape[0] >= 2 and y.shape[0] >= 2:
                    terms.append(module._two_sample_mmd(x, y))
        return torch.stack(terms).mean() if terms else value.sum() * 0.0

    slots = []
    if module.background_only_training:
        slots.append(across_batches(flat(bg["qz_m"]), bg_batch))
        if module.n_spatial_latent > 0:
            slots.append(across_batches(flat(bg["qsp_m"]), bg_batch))
    else:
        combined_batch = torch.cat([bg_batch, tg_batch])
        slots.append(across_batches(
            torch.cat([flat(bg["qz_m"]), flat(tg["qz_m"])]), combined_batch
        ))
        if module.n_spatial_latent > 0:
            slots.append(across_batches(
                torch.cat([flat(bg["qsp_m"]), flat(tg["qsp_m"])]), combined_batch
            ))
        # Salient outputs are semantically zero on controls, so align only the
        # real target-arm posteriors rather than comparing against a zero block.
        slots.append(across_batches(flat(tg["qs_m"]), tg_batch))
        if module.n_spatial_latent > 0 and module.spatial_attach == "both":
            slots.append(across_batches(flat(tg["qssp_m"]), tg_batch))
    return torch.stack(slots).mean()


def install_four_slot_batch_mmd(module, weight: float):
    """Add an annealed slice-MMD term to a constructed Haruka module instance."""
    if weight < 0:
        raise ValueError("batch MMD weight must be non-negative")
    if getattr(module, "_four_slot_batch_mmd_installed", False):
        raise RuntimeError("four-slot batch MMD is already installed")
    original_loss = module.loss

    def loss_with_batch_mmd(
        self, concat_tensors, inference_outputs, generative_outputs, kl_weight=1.0
    ):
        output = original_loss(
            concat_tensors, inference_outputs, generative_outputs, kl_weight
        )
        if weight > 0 and self.n_batch > 1:
            penalty = _four_slot_batch_mmd(
                self, inference_outputs, concat_tensors
            )
            scaled = self._anneal_weight() * weight * penalty
            object.__setattr__(output, "loss", output.loss + scaled)
            output.extra_metrics["four_slot_batch_mmd"] = penalty.detach()
            output.extra_metrics["four_slot_batch_mmd_weight"] = torch.as_tensor(
                weight, device=penalty.device, dtype=penalty.dtype
            )
        return output

    module.loss = MethodType(loss_with_batch_mmd, module)
    module._four_slot_batch_mmd_installed = True
    module.four_slot_batch_mmd_weight = float(weight)
    return module
