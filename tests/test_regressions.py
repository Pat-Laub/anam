import numpy as np
import pandas as pd
import pytest
import torch

from anam import ANAM, SparsityConfig


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


class TestRefit:
    @pytest.mark.parametrize("chosen", [False, True])
    def test_refit_matches_fresh_fit(self, df, chosen):
        X, y = df[["x", "z"]], df["y"]
        kwargs = dict(epochs=3, random_state=0, verbose=False)

        def make(X):
            model = ANAM() if chosen else nam()
            if chosen:
                model.rank_main_effects(
                    X, y, ensemble_size=1, ensemble_epochs=1, random_state=0,
                    verbose=False,
                )
                model.choose(n_main=2, n_interactions=0)
            return model

        refit = make(X).fit(X, y, **kwargs).fit(X + 100, y, **kwargs)
        fresh = make(X + 100).fit(X + 100, y, **kwargs)
        np.testing.assert_allclose(
            refit.predict(X + 100), fresh.predict(X + 100), rtol=1e-3
        )

    def test_auto_selection_reruns(self, monkeypatch):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=["x0", "x1", "x2"])
        monkeypatch.setattr(
            "anam.api.SparsityConfig",
            lambda **k: SparsityConfig(
                **{**k, "num_ensemble": 1, "ensemble_epochs": 100, "batch_size": 100}
            ),
        )
        model = ANAM(interactions=[])
        for name in ["x0", "x2"]:
            y = pd.Series(rng.poisson(np.exp(X[name])))
            model.fit(X, y, epochs=2, random_state=0, verbose=False)
            assert model.main_importance_.index[0] == name


class TestConfigErrors:
    @pytest.mark.parametrize(
        "monotone", [{"typo": "increasing"}, {"x": "decreasng"}, {"c": "increasing"}]
    )
    def test_bad_monotone(self, df, monotone):
        with pytest.raises(ValueError, match="monotone"):
            nam(monotone=monotone).fit(df[["x", "c"]], df["y"], epochs=1, verbose=False)

    def test_bad_monotone_when_ranking(self, df):
        with pytest.raises(ValueError, match="monotone"):
            ANAM(monotone={"typo": "increasing"}).rank_main_effects(
                df[["x", "z"]], df["y"], ensemble_size=1, ensemble_epochs=1,
                verbose=False,
            )

    def test_missing_exposure_in_fit(self, df):
        with pytest.raises(ValueError, match="expoo"):
            nam(exposure="expoo").fit(
                df[["x", "expo"]], df["y"], epochs=1, verbose=False
            )

    def test_missing_exposure_in_predict(self, df):
        model = nam(exposure="expo").fit(
            df[["x", "expo"]], df["y"], epochs=1, verbose=False
        )
        with pytest.raises(ValueError, match="expo"):
            model.predict(df[["x"]])
