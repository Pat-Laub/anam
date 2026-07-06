"""
Tests for the user-facing ANAM wrapper (api.py).

Covers construct / fit / predict on raw DataFrames, with lazy build,
internal preprocessing, exposure offset handling, and the validation-split path.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from anam import ANAM
from anam.model import ANAMModule


@pytest.fixture
def gamma_df():
    """Synthetic Gamma regression frame with a clear signal."""
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    eta = 1.0 + 1.2 * x1 - 0.8 * x2
    mu = np.exp(eta)
    y = rng.gamma(shape=10.0, scale=mu / 10.0)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def belgian_like_df():
    """Belgian-like frame: string categorical, exposure column, target in X."""
    rng = np.random.default_rng(1)
    n = 4000
    ageph = rng.uniform(18, 80, n)
    power = rng.uniform(20, 200, n)
    fuel = rng.choice(["petrol", "diesel"], n)
    bm = rng.integers(0, 20, n).astype(float)
    expo = rng.uniform(0.1, 1.0, n)
    eta = -2.0 + 0.01 * (ageph - 40) + 0.4 * bm / 10 + (fuel == "diesel") * 0.3
    mu = np.exp(eta) * expo
    nclaims = rng.poisson(mu).astype(float)
    return pd.DataFrame(
        {
            "ageph": ageph,
            "power": power,
            "fuel": fuel,
            "bm": bm,
            "expo": expo,
            "nclaims": nclaims,
        }
    )


def _intercept_baseline_loss(y_train, y_test):
    """Deviance-style baseline: predict the training mean everywhere."""
    mean = np.mean(y_train)
    return np.mean((y_test - mean) ** 2)


class TestConstructor:
    def test_no_work_in_init(self):
        model = ANAM(distribution="gamma", features="all", interactions=[])
        assert model.module_ is None
        assert model.distribution == "gamma"

class TestFitPredictGamma:
    def test_fit_predict_beats_intercept(self, gamma_df):
        train = gamma_df.iloc[:2400]
        test = gamma_df.iloc[2400:]
        model = ANAM(distribution="gamma", features="all", interactions=[])
        X_train = train.drop("y", axis=1)
        y_train = train["y"]
        model.fit(
            X_train, y_train, epochs=200, lr=0.05, patience=25,
            random_state=0, verbose=False,
        )
        pred = model.predict(test)
        assert pred.shape == (len(test),)
        model_mse = np.mean((test["y"].values - pred) ** 2)
        baseline = _intercept_baseline_loss(train["y"].values, test["y"].values)
        assert model_mse < baseline

    def test_predict_needs_only_raw_X(self, gamma_df):
        model = ANAM(distribution="gamma", features="all", interactions=[])
        model.fit(gamma_df.drop("y", axis=1), gamma_df["y"], epochs=30, random_state=0, verbose=False)
        # No arguments beyond raw X.
        pred = model.predict(gamma_df)
        assert np.all(np.isfinite(pred))

    def test_builds_module(self, gamma_df):
        model = ANAM(distribution="gamma", features="all", interactions=[])
        model.fit(gamma_df.drop("y", axis=1), gamma_df["y"], epochs=5, random_state=0, verbose=False)
        assert isinstance(model.module_, ANAMModule)
        assert set(model.feature_names_) == {"x1", "x2"}


class TestBelgianLike:
    def test_string_categorical_and_exposure(self, belgian_like_df):
        train = belgian_like_df.iloc[:3200]
        test = belgian_like_df.iloc[3200:]
        model = ANAM(
            distribution="poisson",
            exposure="expo",
            monotone={"bm": "increasing"},
            features="all",
            interactions=[],
        )
        # y passed whole; target column present in X should be dropped.
        X_train = train.drop("nclaims", axis=1)
        y_train = train["nclaims"]
        model.fit(
            X_train, y_train, epochs=150, lr=0.05, patience=20,
            random_state=2000, verbose=False,
        )
        # Target and exposure columns excluded from features.
        assert "nclaims" not in model.feature_names_
        assert "expo" not in model.feature_names_
        assert "fuel" in model.categorical_features_
        pred = model.predict(test)
        assert pred.shape == (len(test),)
        assert np.all(pred >= 0)

    def test_monotone_bm_non_decreasing(self, belgian_like_df):
        model = ANAM(
            distribution="poisson",
            exposure="expo",
            monotone={"bm": "increasing"},
            features="all",
            interactions=[],
        )
        X = belgian_like_df.drop("nclaims", axis=1)
        y = belgian_like_df["nclaims"]
        model.fit(
            X, y, epochs=150, lr=0.05, patience=20, random_state=2000, verbose=False,
        )
        # Evaluate the bm shape function on a standardized grid (monotone lattice).
        scaler = model.scalers_["bm"]
        grid_raw = np.linspace(0, 19, 40).reshape(-1, 1)
        grid_std = torch.tensor(
            scaler.transform(grid_raw).ravel(), dtype=torch.float32
        )
        model.module_.eval()
        with torch.no_grad():
            vals = model.module_.get_shape_function("bm", grid_std).numpy()
        assert np.all(np.diff(vals) >= -1e-4)


class TestValidationSplit:
    def test_validation_split_runs_without_X_val(self, gamma_df):
        model = ANAM(distribution="gamma", features="all", interactions=[])
        model.fit(
            gamma_df.drop("y", axis=1), gamma_df["y"], validation_split=0.2,
            epochs=50, random_state=0, verbose=False,
        )
        assert model.history_ is not None
        assert len(model.history_["val_loss"]) > 0

    def test_validation_split_zero_no_val(self, gamma_df):
        model = ANAM(distribution="gamma", features="all", interactions=[])
        model.fit(
            gamma_df.drop("y", axis=1), gamma_df["y"], validation_split=0.0,
            epochs=20, random_state=0, verbose=False,
        )
        assert len(model.history_["val_loss"]) == 0


@pytest.fixture
def fitted_belgian(belgian_like_df):
    """A quickly-fitted Poisson/exposure model with one interaction."""
    model = ANAM(
        distribution="poisson",
        exposure="expo",
        monotone={"bm": "increasing"},
        features="all",
        interactions=[("ageph", "power")],
    )
    X = belgian_like_df.drop("nclaims", axis=1)
    y = belgian_like_df["nclaims"]
    model.fit(
        X, y, epochs=60, lr=0.05, patience=15, random_state=2000, verbose=False,
    )
    return model, belgian_like_df


class TestPredictComponents:
    def test_components_reproduce_linear_predictor(self, fitted_belgian):
        model, df = fitted_belgian
        comps = model.predict_components(df)
        recon = comps.sum(axis=1).values
        log_pred = np.log(model.predict(df))
        assert np.allclose(recon, log_pred, atol=1e-4)

    def test_components_has_bias_offset_and_terms(self, fitted_belgian):
        model, df = fitted_belgian
        comps = model.predict_components(df)
        assert "bias" in comps.columns
        assert "offset" in comps.columns
        for name in model.feature_names_:
            assert name in comps.columns
        assert "ageph_power" in comps.columns
        # Offset is log(expo) for this exposure model.
        assert np.allclose(comps["offset"].values, np.log(df["expo"].values))


class TestEffect:
    def test_continuous_effect_grid_endpoints(self, fitted_belgian):
        model, df = fitted_belgian
        grid, values = model.effect("ageph")
        assert grid.shape == values.shape
        assert np.isclose(grid.min(), df["ageph"].min(), atol=1e-6)
        assert np.isclose(grid.max(), df["ageph"].max(), atol=1e-6)

    def test_categorical_effect_returns_labels(self, fitted_belgian):
        model, _ = fitted_belgian
        labels, values = model.effect("fuel")
        labels = [str(lbl) for lbl in labels]
        assert set(labels) == {"petrol", "diesel"}
        assert len(values) == len(labels)

    def test_interaction_effect_shapes(self, fitted_belgian):
        model, _ = fitted_belgian
        grid_a, grid_b, vals = model.effect("ageph", "power")
        assert vals.shape == (len(grid_a), len(grid_b))

    def test_interaction_effect_order_is_transposed(self, fitted_belgian):
        """effect(a, b) and effect(b, a) must be transposes in original units.

        The interaction is stored under a single key ("ageph_power"); querying
        the reverse order drives the wrapper's transpose compensation. This
        regression test pins that effect(b, a) returns exactly effect(a, b).T,
        with the axes swapped to match — so the compensation and the engine's
        canonical-order behaviour stay in agreement.
        """
        model, _ = fitted_belgian
        grid_a, grid_b, vals_ab = model.effect("ageph", "power")
        grid_b2, grid_a2, vals_ba = model.effect("power", "ageph")

        # Axes come back in the requested order.
        assert np.allclose(grid_a, grid_a2)
        assert np.allclose(grid_b, grid_b2)
        # And the surface is the transpose.
        assert vals_ba.shape == (vals_ab.shape[1], vals_ab.shape[0])
        assert np.allclose(vals_ba, vals_ab.T, atol=1e-6)


class TestImportance:
    def test_importance_series_covers_all_terms(self, fitted_belgian):
        model, _ = fitted_belgian
        imp = model.importance()
        assert isinstance(imp, pd.Series)
        expected = set(model.feature_names_) | {"ageph_power"}
        assert set(imp.index) == expected
        # Sorted descending.
        assert list(imp.values) == sorted(imp.values, reverse=True)


class TestPlottingSmoke:
    def test_plot_methods_headless(self, fitted_belgian):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model, _ = fitted_belgian
        ax = model.plot_effect("ageph")  # continuous
        assert ax is not None
        ax = model.plot_effect("fuel")  # categorical
        assert ax is not None
        ax = model.plot_effect("ageph", "power")  # interaction heatmap
        assert ax is not None
        fig = model.plot_effects()
        assert fig is not None
        fig = model.plot_effects(["ageph", "fuel", ("ageph", "power")])
        assert fig is not None
        ax = model.plot_importance()
        assert ax is not None
        plt.close("all")


class TestPersistence:
    def test_save_load_round_trip_identical_predictions(self, fitted_belgian, tmp_path):
        model, df = fitted_belgian
        original = model.predict(df)

        path = tmp_path / "anam_model.pt"
        model.save(path)
        loaded = ANAM.load(path)

        restored = loaded.predict(df)
        assert np.array_equal(original, restored)

    def test_loaded_model_interpretation_without_refit(self, fitted_belgian, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model, _ = fitted_belgian
        path = tmp_path / "anam_model.pt"
        model.save(path)
        loaded = ANAM.load(path)

        # importance() returns the same Series values without recomputation.
        orig_imp = model.importance()
        loaded_imp = loaded.importance()
        assert list(loaded_imp.index) == list(orig_imp.index)
        assert np.array_equal(loaded_imp.values, orig_imp.values)

        # plot_effects() renders headlessly without a refit.
        fig = loaded.plot_effects()
        assert fig is not None
        plt.close("all")

    def test_load_legacy_state_dict_raises(self, fitted_belgian, tmp_path):
        model, _ = fitted_belgian
        path = tmp_path / "legacy.pt"
        # Legacy format: a raw module state_dict, not the v0.2 bundle.
        torch.save(model.module_.state_dict(), path)
        with pytest.raises((RuntimeError, ValueError)):
            ANAM.load(path)

    def test_save_before_fit_raises(self, tmp_path):
        model = ANAM(distribution="gamma", features="all", interactions=[])
        with pytest.raises(RuntimeError):
            model.save(tmp_path / "unfitted.pt")
