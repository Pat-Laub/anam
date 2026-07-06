"""
Variable-selection tests for the wrapper.

The selection surface is three explicit steps — ``rank_main_effects`` (stage 1),
``rank_interactions`` (stage 2, over the chosen mains, strong heredity) and the
cheap dataless ``choose`` commit — plus the fully-automatic ``fit`` path and the
selection plots.

Kept minutes-fast with a small screening config (few ensemble members, few
epochs) on the package's synthetic generator.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import anam.sparsity as sparsity
from anam import ANAM
from anam.utils import generate_synthetic_data


# True structure of generate_synthetic_data: mains X1,X2; interactions
# (X3,X4),(X5,X6),(X7,X8); X9,X10 are noise.
TRUE_INTERACTIONS = {("X3", "X4"), ("X5", "X6"), ("X7", "X8")}

# Fast screening knobs shared by the ranking helpers.
FAST = dict(ensemble_size=3, ensemble_epochs=60, ensemble_patience=12,
            random_state=0, verbose=False)


@pytest.fixture(scope="module")
def synth_df():
    return generate_synthetic_data(n_samples=8000, noise_level=0.1, random_state=0)


@pytest.fixture(scope="module")
def clean_df():
    """Easy linear mains + two strong bilinear interactions, X7/X8 noise.

    All interaction features carry a linear main effect (so strong heredity
    keeps them) and the mains are easy to fit, so stage 2 is unconfounded and
    cleanly ranks the true pairs (X3,X4) and (X5,X6) on top.
    """
    rng = np.random.default_rng(0)
    n = 6000
    X = {f"X{i}": rng.uniform(-1, 1, n) for i in range(1, 9)}
    eta = (
        0.5 + 0.8 * X["X1"] - 0.6 * X["X2"]
        + 0.5 * X["X3"] + 0.5 * X["X4"] + 0.5 * X["X5"] + 0.5 * X["X6"]
        + 2.0 * X["X3"] * X["X4"] + 2.0 * X["X5"] * X["X6"]
    )
    mu = np.exp(eta)
    shape = 50.0  # low dispersion -> clean signal
    y = rng.gamma(shape, mu / shape)
    df = pd.DataFrame(X)
    df["Y"] = y
    return df


CLEAN_INTERACTIONS = {frozenset(("X3", "X4")), frozenset(("X5", "X6"))}


def _rank(model, df, n_main="auto", **kwargs):
    """Run both ranking stages with a small, fast screening config."""
    screen = {**FAST, **kwargs}
    X = df.drop("Y", axis=1)
    y = df["Y"]
    model.rank_main_effects(X, y, **screen)
    model.rank_interactions(X, y, n_main=n_main, **screen)
    return model


def _select(model, df, n_main="auto", n_interactions="auto", **kwargs):
    """Rank both stages, then commit via choose."""
    _rank(model, df, n_main=n_main, **kwargs)
    model.choose(n_main=n_main, n_interactions=n_interactions)
    return model


class TestAutoSelectionRecoversStructure:
    def test_auto_ranks_noise_last_and_drops_it(self, synth_df):
        model = ANAM(distribution="gamma")  # both "auto"
        _select(model, synth_df)

        # The two noise features rank at the bottom of the importance table...
        bottom_two = set(model.main_importance_.index[-2:])
        assert bottom_two == {"X9", "X10"}
        # ...and the elbow never keeps them.
        assert "X9" not in model.features_ and "X10" not in model.features_

    def test_recovers_true_interactions(self, clean_df):
        # On the unconfounded dataset stage 2 ranks the two true bilinear pairs
        # at the very top, and auto selection recovers exactly them.
        model = ANAM(distribution="gamma")
        _select(
            model, clean_df, n_main=6, n_interactions="auto",
            ensemble_epochs=120, ensemble_patience=20,
        )

        top2 = {frozenset(p) for p in model.interaction_scores_.index[:2]}
        assert top2 == CLEAN_INTERACTIONS, (
            f"expected the true pairs on top; got {list(top2)}"
        )
        # Auto interaction count recovers exactly the two true pairs.
        selected = {frozenset(p) for p in model.interactions_}
        assert selected == CLEAN_INTERACTIONS

    def test_attributes_are_series(self, synth_df):
        model = ANAM(distribution="gamma")
        _select(model, synth_df)
        assert isinstance(model.main_importance_, pd.Series)
        assert isinstance(model.interaction_scores_, pd.Series)
        assert model.baseline_val_loss_ is not None


class TestHeredity:
    def test_interactions_are_within_chosen_mains(self, clean_df):
        # rank_interactions only screens pairs of the top-n_main mains, so every
        # ranked pair references features in that heredity basis.
        model = ANAM(distribution="gamma")
        _rank(model, clean_df, n_main=4, ensemble_epochs=120, ensemble_patience=20)

        assert len(model._interaction_basis_) == 4
        basis = set(model._interaction_basis_)
        feats_in_pairs = {f for pair in model.interaction_scores_.index for f in pair}
        assert feats_in_pairs <= basis


class TestChoose:
    def test_choose_before_ranking_interactions_raises(self, synth_df):
        model = ANAM(distribution="gamma")
        X = synth_df.drop("Y", axis=1)
        y = synth_df["Y"]
        model.rank_main_effects(X, y, **FAST)
        # Mains ranked but interactions not; asking for any interactions errors.
        with pytest.raises(RuntimeError):
            model.choose(n_main=4, n_interactions=2)

    def test_choose_without_interactions_needs_no_ranking(self, synth_df):
        model = ANAM(distribution="gamma")
        X = synth_df.drop("Y", axis=1)
        y = synth_df["Y"]
        model.rank_main_effects(X, y, **FAST)
        # n_interactions=0 is a pure main-effects commit -> no stage 2 needed.
        model.choose(n_main=4, n_interactions=0)
        assert len(model.features_) == 4
        assert model.interactions_ == []

    def test_choose_n_main_mismatch_raises(self, clean_df):
        model = ANAM(distribution="gamma")
        _rank(model, clean_df, n_main=6, ensemble_epochs=120, ensemble_patience=20)
        # Interactions were ranked over 6 mains; committing 4 would violate the
        # heredity the pairs were screened under.
        with pytest.raises(ValueError):
            model.choose(n_main=4, n_interactions=1)


class TestCaching:
    def test_stage1_not_retrained_on_second_rank(self, synth_df, monkeypatch):
        calls = {"n": 0}
        real = sparsity.train_ensemble_main_effects

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(sparsity, "train_ensemble_main_effects", counting)

        model = ANAM(distribution="gamma")
        _select(model, synth_df, n_main=6)
        _select(model, synth_df, n_main=4)  # different n_main

        assert calls["n"] == 1, "stage-1 ensemble should be cached across calls"

    def test_changing_n_interactions_reslices(self, synth_df, monkeypatch):
        model = ANAM(distribution="gamma")
        _select(model, synth_df, n_main=6, n_interactions="auto")

        calls = {"n": 0}
        real = sparsity.evaluate_candidate_interactions

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(sparsity, "evaluate_candidate_interactions", counting)
        # Same mains, only the commit count changes -> no stage-2 recompute.
        _rank(model, synth_df, n_main=6)
        model.choose(n_main=6, n_interactions=1)
        assert calls["n"] == 0


class TestPlots:
    def test_plot_interaction_importance_returns_axis(self, clean_df):
        model = ANAM(distribution="gamma")
        _rank(model, clean_df, n_main=6, ensemble_epochs=120, ensemble_patience=20)
        ax = model.plot_interaction_importance(n_interactions=2)
        assert ax is not None

    def test_plot_main_effect_importance_returns_axis(self, synth_df):
        model = ANAM(distribution="gamma")
        X = synth_df.drop("Y", axis=1)
        y = synth_df["Y"]
        model.rank_main_effects(X, y, **FAST)
        ax = model.plot_main_effect_importance(n_main=5)
        assert ax is not None


class TestOptOut:
    def test_interactions_all_builds_all_pairs(self, synth_df):
        feats = ["X1", "X2", "X3", "X4"]
        model = ANAM(distribution="gamma", features=feats, interactions="all")
        model.fit(
            synth_df.drop("Y", axis=1), synth_df["Y"], epochs=20, lr=0.05, patience=10,
            random_state=0, verbose=False,
        )
        # 4 features -> C(4,2) = 6 interaction subnets.
        assert len(model.module_.interaction_subnets) == 6


class TestFitComposition:
    def test_fit_alone_triggers_selection(self, synth_df):
        # Small frame so default-ish selection stays fast.
        small = generate_synthetic_data(n_samples=1500, noise_level=0.1, random_state=1)
        model = ANAM(distribution="gamma")
        # Pre-run selection with a small config, then fit respects it.
        _select(model, small)
        model.fit(
            small.drop("Y", axis=1), small["Y"], epochs=30, lr=0.05, patience=10,
            random_state=0, verbose=False,
        )
        assert model.features_ is not None
        assert model.interactions_ is not None
        # Module built only on the selected terms.
        assert set(model.feature_names_) == set(model.features_)
        fig = model.plot_selection()
        assert fig is not None

    def test_fit_respects_manual_choose(self, synth_df):
        # A manual choose() must not be overridden by a later bare fit().
        model = ANAM(distribution="gamma")
        _rank(model, synth_df, n_main=5)
        model.choose(n_main=5, n_interactions=0)
        chosen = list(model.features_)
        model.fit(
            synth_df.drop("Y", axis=1), synth_df["Y"], X_val=synth_df.drop("Y", axis=1), y_val=synth_df["Y"],
            epochs=15, lr=0.05, patience=8, random_state=0, verbose=False,
        )
        assert list(model.features_) == chosen
        assert model.interactions_ == []

    def test_validation_split_zero_auto_raises(self, synth_df):
        model = ANAM(distribution="gamma")
        with pytest.raises(ValueError):
            model.fit(
                synth_df.drop("Y", axis=1), synth_df["Y"], validation_split=0.0,
                epochs=5, verbose=False,
            )
