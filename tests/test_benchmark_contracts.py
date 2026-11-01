"""Regression tests for simulator and benchmark metadata contracts."""

from __future__ import annotations

import os
import sys
import unittest

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench_merfish"))

from build_sim_v4 import (
    _bilinear_interaction,
    _salient_state_probabilities,
    _stable_seed,
)
from run_baselines_2x2 import BATCH_KEY as BASELINE_BATCH_KEY
from orthogonalize_haruka import residualize
from run_haruka import SHARED, VARIANTS, validate_batch_condition


class BenchmarkContractTest(unittest.TestCase):
    def test_simulator_seed_is_stable(self):
        self.assertEqual(_stable_seed(0, "T1", "control"), 1443197456)
        self.assertEqual(_stable_seed(7, "T4", "disease"), 125456058)

    def test_bilinear_coherence_preserves_strict_interaction_contract(self):
        import numpy as np

        rng = np.random.default_rng(11)
        background = rng.normal(size=(8, 16))
        salient = rng.normal(size=(8, 16))

        np.testing.assert_allclose(
            _bilinear_interaction(background, salient, 0.0),
            background * salient,
        )
        expected_coherent = (
            np.sqrt(np.mean(background ** 2, axis=1, keepdims=True)) * salient
        )
        np.testing.assert_allclose(
            _bilinear_interaction(background, salient, 1.0), expected_coherent
        )
        for coherence in (0.0, 0.35, 1.0):
            np.testing.assert_allclose(
                _bilinear_interaction(np.zeros_like(background), salient, coherence),
                0.0,
            )
            np.testing.assert_allclose(
                _bilinear_interaction(background, np.zeros_like(salient), coherence),
                0.0,
            )
        mixed_carrier = _bilinear_interaction(
            background, np.ones_like(background), 0.35
        )
        np.testing.assert_allclose(
            np.mean(mixed_carrier ** 2, axis=1),
            np.mean(background ** 2, axis=1),
        )

    def test_independent_salient_state_has_identical_distribution_in_each_domain(self):
        import numpy as np

        domain = np.repeat(np.arange(3), 5)
        probability = _salient_state_probabilities(
            domain,
            n_states=2,
            q_ss=np.asarray([0.1, 0.5, 0.9]),
            p_ss=None,
            independent=True,
            independent_binary_rate=0.5,
        )
        np.testing.assert_array_equal(probability, np.full(len(domain), 0.5))

        legacy = _salient_state_probabilities(
            domain,
            n_states=2,
            q_ss=np.asarray([0.1, 0.5, 0.9]),
            p_ss=None,
        )
        np.testing.assert_array_equal(legacy, np.repeat([0.1, 0.5, 0.9], 5))

    def test_baseline_batch_is_condition_matched(self):
        self.assertEqual(BASELINE_BATCH_KEY, "timepoint")
        self.assertNotEqual(BASELINE_BATCH_KEY, "slice_id")

    def test_mae_ablation_has_true_zero_and_selected_default(self):
        for epochs in (0, 30, 60, 90):
            cfg = VARIANTS[f"both_sal8_timebatch_mae{epochs}"]
            self.assertEqual(cfg["mae_epochs"], epochs)
            self.assertEqual(cfg["batch_key"], "timepoint")
        default = VARIANTS["both_sal8_timebatch"]
        self.assertEqual(default["mae_epochs"], 90)
        self.assertEqual(default["paired_slot_reg_weight"], 1.0)
        self.assertEqual(default["paired_slot_reg_start_epoch"], 90)
        self.assertEqual(default["background_representation"], "private")
        self.assertEqual(default["n_hidden"], 256)

    def test_end_to_end_cleanup_variants_are_raw_slot_regularizers(self):
        for weight in (1, 5, 20):
            cfg = VARIANTS[f"both_sal8_timebatch_reg{weight}"]
            self.assertEqual(cfg["paired_slot_reg_weight"], float(weight))
            self.assertEqual(cfg["batch_key"], "timepoint")
            self.assertEqual(cfg["spatial_attach"], "both")

    def test_private_background_variant_excludes_gat_carrier(self):
        cfg = VARIANTS["both_sal8_timebatch_stage_reg1_private"]
        self.assertEqual(cfg["background_representation"], "private")
        self.assertLess(cfg["n_spatial_latent"], 32)

    def test_background_aligned_training_has_equal_pair_weights_and_capacity(self):
        cfg = VARIANTS["both_bg_aligned"]
        self.assertEqual(cfg["latent_dim"], cfg["salient_latent_dim"])
        self.assertLess(cfg["n_spatial_latent"], cfg["latent_dim"])
        background_pair_weight = cfg.get("lam_mi", SHARED["lam_mi"])
        salient_pair_weight = background_pair_weight * cfg["salient_within_penalty"]
        self.assertEqual(background_pair_weight, 5.0)
        self.assertEqual(salient_pair_weight, background_pair_weight)
        self.assertEqual(VARIANTS["both_fully_symmetric"], cfg)

    def test_clean_cvi_variant_has_only_symmetric_within_pair_regularizers(self):
        cfg = {**SHARED, **VARIANTS["both_clean_cvi_symmetric"]}
        self.assertEqual(cfg["latent_dim"], cfg["salient_latent_dim"])
        self.assertTrue(cfg["condition_only_salient_spatial"])
        self.assertEqual(cfg["wasserstein_penalty"], 0.2)
        self.assertEqual(cfg["lam_mi"], 5.0)
        self.assertEqual(cfg["salient_within_penalty"], 1.0)
        self.assertEqual(cfg["spatial_pair_penalty"], 0.0)
        self.assertEqual(cfg["spatial_invariance_penalty"], 0.0)
        self.assertEqual(cfg["paired_slot_reg_weight"], 0.0)
        self.assertEqual(cfg["background_mmd_penalty"], 0.0)
        self.assertFalse(cfg["penalize_salient_spatial"])
        self.assertFalse(cfg["mask_salient_mean_on_background"])
        self.assertGreater(cfg["mae_epochs"], 0)

    def test_capacity_and_training_variants_change_one_axis(self):
        base = VARIANTS["both_sal8_timebatch_stage_reg1_private"]
        long = VARIANTS["both_sal8_timebatch_stage_reg1_private_long"]
        wide = VARIANTS["both_sal16_timebatch_stage_reg1_private"]
        hidden = VARIANTS["both_sal8_timebatch_stage_reg1_private_hidden256"]
        self.assertEqual(base["mae_epochs"], 90)
        self.assertEqual(long["mae_epochs"], 150)
        self.assertEqual(long["paired_slot_reg_start_epoch"], 150)
        self.assertEqual(wide["latent_dim"], 64)
        self.assertEqual(wide["n_spatial_latent"], 16)
        self.assertEqual(hidden["n_hidden"], 256)

    def test_staged_cleanup_starts_after_mae(self):
        for weight in (1, 5, 20):
            cfg = VARIANTS[f"both_sal8_timebatch_stage_reg{weight}"]
            self.assertEqual(cfg["mae_epochs"], 90)
            self.assertEqual(cfg["paired_slot_reg_start_epoch"], 90)
            self.assertEqual(cfg["paired_slot_reg_ramp_epochs"], 10)

    def test_condition_confounded_batch_is_rejected(self):
        obs = pd.DataFrame(
            {
                "slice_id": ["T1_control", "T1_control", "T1_disease", "T1_disease"],
                "timepoint": ["T1", "T1", "T1", "T1"],
                "condition": ["control", "control", "disease", "disease"],
            }
        )
        with self.assertRaisesRegex(ValueError, "condition-confounded"):
            validate_batch_condition(obs, "slice_id")
        validate_batch_condition(obs, "timepoint")

    def test_residualize_removes_linear_shared_subspace(self):
        import numpy as np

        rng = np.random.default_rng(3)
        nuisance = rng.normal(size=(500, 3))
        private = rng.normal(size=(500, 2))
        target = nuisance @ rng.normal(size=(3, 2)) + private
        cleaned = residualize(target, nuisance, alpha=0.0)
        cross_cov = (nuisance - nuisance.mean(0)).T @ cleaned / len(cleaned)
        np.testing.assert_allclose(cross_cov, 0.0, atol=1e-6)
        self.assertEqual(cleaned.shape, target.shape)


if __name__ == "__main__":
    unittest.main()
