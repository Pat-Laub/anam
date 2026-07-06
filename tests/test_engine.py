"""
Engine-level helper tests:

- ``resolve_distribution`` registry lookup
- ``_find_elbow`` kneedle helper
- offset threading through the sparsity selection stages
"""

import numpy as np
import pandas as pd
import pytest
import torch

from anam.losses import (
    resolve_distribution,
    DISTRIBUTIONS,
    poisson_nll,
    gamma_nll,
    gaussian_nll,
)
from anam.sparsity import _find_elbow, train_ensemble_main_effects, compute_ensemble_importance
from anam.sparsity import SparsityConfig
from anam.utils import build_feature_configs
from anam.optimizer import prepare_data


class TestResolveDistribution:
    def test_poisson(self):
        loss, link = resolve_distribution("poisson")
        assert loss is poisson_nll
        assert link == "exp"

    def test_gamma(self):
        loss, link = resolve_distribution("gamma")
        assert loss is gamma_nll
        assert link == "exp"

    def test_gaussian(self):
        loss, link = resolve_distribution("gaussian")
        assert loss is gaussian_nll
        assert link == "identity"

    def test_unknown_lists_valid_names(self):
        with pytest.raises(ValueError) as exc:
            resolve_distribution("bernoulli")
        msg = str(exc.value)
        for name in DISTRIBUTIONS:
            assert name in msg

    def test_gaussian_nll_is_mse(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        p = torch.tensor([1.5, 2.5, 2.0])
        expected = torch.mean((y - p) ** 2)
        assert gaussian_nll(y, p).item() == pytest.approx(expected.item())


class TestFindElbow:
    def test_monotone_scores_single_kink(self):
        # Sharp drop after the third value -> keep 3.
        scores = [10.0, 9.5, 9.0, 1.0, 0.9, 0.8]
        assert _find_elbow(scores) == 3

    def test_flat_scores_keep_all(self):
        scores = [1.0, 1.0, 1.0, 1.0]
        assert _find_elbow(scores) == 4

    def test_linear_scores_keep_all(self):
        # A perfectly linear decline has no elbow -> keep all.
        scores = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _find_elbow(scores) == 5

    def test_two_candidates_keep_all(self):
        assert _find_elbow([5.0, 1.0]) == 2

    def test_single_candidate(self):
        assert _find_elbow([1.0]) == 1

    def test_ties_keep_more(self):
        # Symmetric profile: two points equidistant from the chord; keep more.
        scores = [10.0, 6.0, 5.0, 4.0, 0.0]
        keep = _find_elbow(scores)
        assert keep >= 3


class TestOffsetThreading:
    def test_ensemble_accepts_offset(self):
        """Stage-1 ensemble runs end-to-end with an offset without error and
        ranks the exposure-driven signal feature above pure noise."""
        rng = np.random.default_rng(0)
        n = 2000
        x1 = rng.uniform(-1, 1, n)      # real signal
        x2 = rng.uniform(-1, 1, n)      # noise
        expo = rng.uniform(0.2, 1.0, n)
        # Rate driven by x1 only; counts scale with exposure.
        mu = np.exp(0.5 + 1.5 * x1) * expo
        y = rng.poisson(mu).astype(float)
        df = pd.DataFrame({"X1": x1, "X2": x2, "nclaims": y, "expo": expo})

        configs = build_feature_configs(df, ["X1", "X2"])
        train_x, train_y, train_off = prepare_data(
            df, ["X1", "X2"], "nclaims", offset="expo"
        )

        cfg = SparsityConfig(
            num_ensemble=2, ensemble_epochs=40, ensemble_patience=10,
            batch_size=1000, verbose=False,
        )
        variances = train_ensemble_main_effects(
            configs, train_x, train_y, train_x, train_y,
            loss_fn=poisson_nll, config=cfg,
            train_offset=train_off, val_offset=train_off,
        )
        importance = compute_ensemble_importance(variances)
        assert importance["X1"] > importance["X2"]
