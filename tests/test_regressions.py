import numpy as np
import pandas as pd
import pytest
import torch

from anam import ANAM


def nam(**kwargs):
    return ANAM(**{"features": "all", "interactions": [], **kwargs})


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "c": rng.choice(["a", "b", "c"], n),
            "expo": rng.uniform(0.1, 1.0, n),
            "y": rng.poisson(0.5, n).astype(float),
        }
    )


class TestInitBias:
    @pytest.mark.parametrize("distribution, shift", [("gamma", 0), ("gaussian", -900)])
    def test_starts_at_mean(self, distribution, shift):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"x": rng.normal(size=500)})
        y = pd.Series(rng.gamma(2, 200, 500) + shift)
        model = nam(distribution=distribution).fit(
            X, y, epochs=1, validation_split=0, verbose=False
        )
        target = np.log(y.mean()) if distribution == "gamma" else y.mean()
        assert model.module_.bias.item() == pytest.approx(target, abs=0.05)

    def test_uses_exposure(self, df):
        model = nam(exposure="expo").fit(
            df[["x", "expo"]], df["y"], epochs=1, validation_split=0, verbose=False
        )
        target = np.log(df["y"].sum() / df["expo"].sum())
        assert model.module_.bias.item() == pytest.approx(target, abs=0.05)

    def test_can_be_disabled(self, df):
        model = nam(init_bias=False).fit(
            df[["x"]], df["y"], epochs=1, validation_split=0, verbose=False
        )
        assert model.module_.bias.item() == pytest.approx(0.0, abs=0.05)

    def test_screening_starts_at_mean(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"x": rng.normal(size=500), "z": rng.normal(size=500)})
        y = pd.Series(rng.gamma(2, 200, 500))
        model = ANAM("gamma").rank_interactions(
            X, y, n_main=2, ensemble_size=1, ensemble_epochs=1,
            random_state=0, verbose=False,
        )
        assert model.baseline_val_loss_ < 10


class TestUnseenCategory:
    def test_has_zero_effect(self, df):
        model = nam(interactions=[("x", "c")]).fit(
            df[["x", "c"]], df["y"], epochs=30, random_state=0, verbose=False
        )
        unseen = pd.DataFrame({"x": [0.5, -1.0], "c": ["new", "new"]})
        components = model.predict_components(unseen)
        assert (components["c"] == 0).all()
        assert (components["x_c"] == 0).all()
        seen = model.predict_components(unseen.assign(c="a"))
        assert (seen["c"] != 0).all() and (seen["x_c"] != 0).all()
