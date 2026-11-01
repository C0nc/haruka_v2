"""Regression tests for the count/COVET boundary in Haruka.

Run in the `haruka2` environment from the repository root::

    python -m unittest tests.test_count_data_contract
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import anndata as ad
import numpy as np
import scipy.sparse
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


class CountDataContractTest(unittest.TestCase):
    def test_condition_only_salient_gat_preserves_cvi_control_posterior(self):
        from contrastivevi_patch._module import ContrastiveVAE

        module = ContrastiveVAE(
            n_input=5,
            n_input_full=5,
            n_batch=1,
            n_hidden=16,
            n_background_latent=4,
            n_salient_latent=4,
            n_spatial_latent=2,
            spatial_attach="both",
            condition_only_salient_spatial=True,
            gene_likelihood="poisson",
        ).eval()
        rng = np.random.default_rng(0)
        module.attach_neighbor_bank(
            rng.poisson(2, size=(12, 5)), np.zeros(12, dtype=np.int64)
        )

        salient_gat_rows = []
        hook = module.gat_mean_salient.register_forward_hook(
            lambda _module, inputs, _output: salient_gat_rows.append(inputs[0].shape[0])
        )

        def arm(offset):
            return {
                "x": torch.tensor(
                    rng.poisson(2, size=(3, 5)), dtype=torch.float32
                ),
                "batch_index": torch.zeros((3, 1), dtype=torch.long),
                "neighbor_idx": (
                    torch.tensor(
                        [[offset, offset + 1],
                         [offset + 1, offset + 2],
                         [offset + 2, offset + 3]]
                    ) % 12
                ),
                "neighbor_mask": torch.ones((3, 2)),
            }

        try:
            with torch.inference_mode():
                output = module.inference(arm(0), arm(5))
        finally:
            hook.remove()

        background = output["background"]
        self.assertEqual(salient_gat_rows, [3])
        self.assertGreater(torch.count_nonzero(background["qs_m"]).item(), 0)
        self.assertEqual(torch.count_nonzero(background["s"]).item(), 0)
        self.assertEqual(torch.count_nonzero(background["qssp_m"]).item(), 0)
        self.assertEqual(torch.count_nonzero(background["s_sp"]).item(), 0)
        torch.testing.assert_close(
            background["qssp_v"], torch.ones_like(background["qssp_v"])
        )

    def test_spatial_setup_seed_resets_lightning_and_scvi(self):
        import haruka_spatial

        previous = haruka_spatial.scvi.settings.seed
        try:
            with mock.patch.object(haruka_spatial, "seed_everything") as reset:
                haruka_spatial._reset_training_seed(17)
            reset.assert_called_once_with(17, workers=True)
            self.assertEqual(haruka_spatial.scvi.settings.seed, 17)
        finally:
            haruka_spatial.scvi.settings.seed = previous

    def test_pure_gat_default_skips_covet(self):
        import haruka_spatial

        counts = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
        adata = ad.AnnData(X=np.log1p(counts))
        adata.layers["counts"] = counts.copy()
        adata.obs["slice_id"] = ["slice", "slice"]
        adata.obsm["spatial"] = np.asarray([[0, 0], [1, 0]], dtype=np.float32)

        with mock.patch.object(haruka_spatial, "_compute_covet") as compute_covet:
            model = haruka_spatial.HarukaSpatial(adata)
            model.construct_cov()

        compute_covet.assert_not_called()
        self.assertEqual(model.cov_gene_num, 0)
        self.assertEqual(model.input_adata.n_vars, adata.n_vars)
        np.testing.assert_array_equal(model.input_adata.layers["counts"], counts)

    def test_covet_augmentation_preserves_raw_counts_layer(self):
        # Import lazily because Haruka's optional scenvi stack is environment-heavy.
        import haruka_spatial

        counts = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        adata = ad.AnnData(X=np.log1p(counts))
        adata.obs["slice_id"] = ["slice", "slice"]
        adata.obs["condition"] = ["control", "disease"]
        adata.layers["counts"] = scipy.sparse.csr_matrix(counts)
        adata.obsm["spatial"] = np.asarray([[0, 0], [1, 0]], dtype=np.float32)
        covet = np.asarray(
            [[[1.0, -0.25], [-0.25, 2.0]], [[3.0, 0.5], [0.5, 4.0]]],
            dtype=np.float32,
        )

        with mock.patch.object(
            haruka_spatial,
            "_compute_covet",
            return_value=(None, covet, None),
        ):
            model = haruka_spatial.HarukaSpatial(adata, cov_gene_num=2)
            model.construct_cov()

        self.assertIn("counts", model.input_adata.layers)
        augmented = np.asarray(model.input_adata.layers["counts"])
        np.testing.assert_array_equal(augmented[:, : counts.shape[1]], counts)
        np.testing.assert_array_equal(augmented[:, counts.shape[1] :], covet.reshape(2, 4))

        # The automatic setup path must select the repaired layer, never fall back
        # to log-normalised X. Mock model registration to keep this a unit test.
        with mock.patch.object(
            haruka_spatial.scvi.external, "ContrastiveVI"
        ) as contrastive_cls:
            model.setup_model(latent_dim=2, n_spatial_latent=0)
        contrastive_cls.setup_anndata.assert_called_once()
        self.assertEqual(model.training_layer_, "counts")

    def test_observed_library_excludes_covet_columns(self):
        from contrastivevi_patch._module import ContrastiveVAE, NeighborGATv2

        gat = NeighborGATv2(3, 2, heads=2, self_query=False).eval()
        neighbors = torch.randn(4, 5, 3)
        self_a = torch.randn(4, 3)
        self_b = torch.randn(4, 3)
        with torch.inference_mode():
            torch.testing.assert_close(
                gat(self_a, neighbors), gat(self_b, neighbors)
            )

        pure_module = ContrastiveVAE(
            n_input=3,
            n_input_full=3,
            n_batch=1,
            n_hidden=8,
            n_background_latent=2,
            n_salient_latent=2,
            dropout_rate=0.0,
            use_observed_lib_size=True,
        )
        self.assertIsNone(pure_module.decoder_me)

        module = ContrastiveVAE(
            n_input=3,
            n_input_full=7,
            n_batch=1,
            n_hidden=8,
            n_background_latent=2,
            n_salient_latent=2,
            dropout_rate=0.0,
            use_observed_lib_size=True,
        ).eval()
        genes = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Deliberately huge/negative COVET values: neither may affect library size.
        covet = torch.tensor([[100.0, 100.0, 100.0, 100.0], [-9.0, -8.0, -7.0, -6.0]])
        x = torch.cat([genes, covet], dim=1)
        with torch.inference_mode():
            out = module._generic_inference(
                x=x,
                index=genes.shape[1],
                batch_index=torch.zeros((2, 1), dtype=torch.long),
            )

        expected = torch.log(genes.sum(1)).unsqueeze(1)
        torch.testing.assert_close(out["library"], expected)


if __name__ == "__main__":
    unittest.main()
