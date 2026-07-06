"""
User-facing ANAM wrapper (the v2 API).

This module defines the ``ANAM`` class that most users interact with: a thin,
``drn``-style wrapper around the raw :class:`~anam.model.ANAMModule` engine. The
wrapper owns preprocessing (standardise continuous columns, ordinal-encode
categoricals, remember the transforms), is built lazily at the first ``fit`` from
the raw DataFrame, maps ``distribution="poisson"`` to a loss + link, treats
exposure as a named column, and plots shape functions in original units.

The engine modules (``model.py``, ``optimizer.py``, ``lattice.py`` ...) are the
computational core and stay untouched; this wrapper only orchestrates them.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import Tensor

from anam import sparsity
from anam.losses import resolve_distribution
from anam.model import ANAMModule
from anam.optimizer import TrainingConfig, train_anam
from anam.sparsity import SparsityConfig


class ANAM:
    """
    Actuarial Neural Additive Model — user-facing wrapper.

    Everything data-independent that defines *what the model is* (distribution,
    constraints, penalties, architecture) is set in the constructor; everything
    about *how long to train* is passed to :meth:`fit`. The model is built
    lazily on the first ``fit`` call, inferring feature types and bounds from the
    training DataFrame.

    Args:
        distribution: ``"poisson"``, ``"gamma"`` or ``"gaussian"``. Sets both the
            loss and the link function (log link for Poisson/Gamma, identity for
            Gaussian). Use ``link=`` to override the link.
        features: ``"auto"`` (default) runs three-stage selection; ``"all"`` uses
            every column except the exposure/target; or an explicit list of names.
        interactions: ``"auto"`` (default), ``"all"`` (every pair), ``[]`` (pure
            main-effects NAM), or an explicit list of ``(a, b)`` tuples.
        categorical: Optional list of column names to force to categorical,
            overriding dtype inference.
        monotone: Dict mapping feature name to ``"increasing"``/``"decreasing"``;
            those features use a monotonic lattice subnet.
        smoothness: Roughness penalty weight applied to all continuous main
            effects (``0.0`` = off).
        marginal_clarity: Orthogonality penalty weight; only active when
            interactions exist.
        exposure: Name of an exposure column used as a Poisson offset. The column
            is excluded from features and (under a log link) logged and added to
            the linear predictor in both ``fit`` and ``predict``.
        num_layers, num_neurons: Main-effect subnet architecture.
        num_layers_interaction, num_neurons_interaction: Interaction subnet
            architecture (default to the main-effect sizes when ``None``).
        num_vertices, num_keypoints: Lattice hyperparameters for monotone terms.
        link: Optional explicit link (``"exp"`` or ``"identity"``) overriding the
            distribution default.
    """

    def __init__(
        self,
        distribution: str = "poisson",
        features="auto",
        interactions="auto",
        categorical: list[str] | None = None,
        monotone: dict[str, str] | None = None,
        smoothness: float = 0.0,
        marginal_clarity: float = 1e-4,
        exposure: str | None = None,
        num_layers: int = 2,
        num_neurons: int = 32,
        num_layers_interaction: int | None = None,
        num_neurons_interaction: int | None = None,
        num_vertices: int = 10,
        num_keypoints: int = 20,
        link: str | None = None,
    ):
        # Store everything verbatim (sklearn convention: no work in __init__).
        self.distribution = distribution
        self.features = features
        self.interactions = interactions
        self.categorical = categorical
        self.monotone = monotone
        self.smoothness = smoothness
        self.marginal_clarity = marginal_clarity
        self.exposure = exposure
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.num_layers_interaction = num_layers_interaction
        self.num_neurons_interaction = num_neurons_interaction
        self.num_vertices = num_vertices
        self.num_keypoints = num_keypoints
        self.link = link

        # Fitted state (populated by _build / fit); trailing underscore = learned.
        self.module_: ANAMModule | None = None
        self.feature_names_: list[str] | None = None
        self.categorical_features_: list[str] = []
        self.continuous_features_: list[str] = []
        self.scalers_: dict = {}
        self.encoders_: dict = {}
        self.cardinalities_: dict = {}
        self.feature_configs_: dict = {}
        self.features_: list[str] | None = None
        self.interactions_: list[tuple[str, str]] | None = None
        self.link_: str | None = None
        self.loss_fn_ = None
        self.history_: dict | None = None
        self.term_importances_ = None
        self.deciles_: dict[str, np.ndarray] | None = None

        # Selection state (populated by rank_*/choose; trailing _ = learned).
        self.main_importance_ = None
        self.interaction_scores_ = None
        self.baseline_val_loss_: float | None = None

        # Internal selection bookkeeping (not part of the public surface).
        self._candidate_feature_configs_: dict | None = None
        self._interaction_losses_: dict | None = None
        # Mains that rank_interactions ranked its candidate pairs against, so
        # choose() can check the requested n_main matches (strong heredity).
        self._interaction_basis_: list[str] | None = None
        self._selection_ran: bool = False
        self._stage1_cache: dict | None = None
        self._stage2_cache: dict | None = None

    # ------------------------------------------------------------------
    # Lazy build + preprocessing
    # ------------------------------------------------------------------
    def _resolve_feature_columns(self, X, y) -> list[str]:
        """Determine which columns of ``X`` are model features."""
        all_cols = list(X.columns)

        exclude = set()
        if self.exposure is not None:
            exclude.add(self.exposure)
        target_name = getattr(y, "name", None)
        if target_name is not None and target_name in all_cols:
            exclude.add(target_name)

        if isinstance(self.features, str):
            if self.features == "all":
                return [c for c in all_cols if c not in exclude]
            if self.features == "auto":
                # Unreachable on the normal path: fit() resolves "auto" through
                # its selection pipeline (rank_main_effects -> choose) and never
                # calls _build/_resolve_feature_columns for it. Only reachable by
                # driving the private build path directly.
                raise RuntimeError(
                    "features='auto' is resolved by fit(); call fit(...) instead "
                    "of building the model directly."
                )
            raise ValueError(
                f"Unknown features={self.features!r}; use 'auto', 'all', or a list."
            )
        # Explicit list: honour it, but still drop exposure/target if present.
        return [c for c in self.features if c not in exclude]

    def _infer_categorical(self, X, feature_cols: list[str]) -> list[str]:
        """Infer categorical columns from dtype, honouring the override."""
        if self.categorical is not None:
            return [c for c in self.categorical if c in feature_cols]

        categorical = []
        for col in feature_cols:
            dtype = X[col].dtype
            # Anything that isn't a numeric dtype is treated as categorical.
            # Note bool is numeric under is_numeric_dtype, so flag it first; and
            # string columns (object, pandas StringDtype/pyarrow-backed str as of
            # pandas 3.0, etc.) must land here rather than being scaled.
            if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(
                dtype
            ):
                categorical.append(col)
        return categorical

    def _candidate_columns(self, X, y) -> list[str]:
        """Return the full candidate pool: every column bar exposure/target.

        Unlike :meth:`_resolve_feature_columns` this ignores ``self.features``
        entirely — it is the universe of columns that ``"auto"``/``"all"``
        selection screens over.
        """
        all_cols = list(X.columns)
        exclude = set()
        if self.exposure is not None:
            exclude.add(self.exposure)
        target_name = getattr(y, "name", None)
        if target_name is not None and target_name in all_cols:
            exclude.add(target_name)
        return [c for c in all_cols if c not in exclude]

    def _fit_preprocessing(self, X, y, feature_cols: list[str]) -> None:
        """Fit scalers/encoders and ``feature_configs`` for ``feature_cols``.

        Sets the loss/link, the categorical/continuous split, the fitted
        transforms and the derived ``feature_configs_`` for exactly the given
        columns. Does NOT resolve interactions or instantiate the module — those
        are the caller's responsibility, so this helper is reusable both for the
        Phase 1–3 direct-build path and the candidate-pool fit done during
        screening (:meth:`_screen_main_effects`).
        """
        # Link / loss from the distribution (with optional link override).
        loss_fn, default_link = resolve_distribution(self.distribution)
        self.loss_fn_ = loss_fn
        self.link_ = self.link if self.link is not None else default_link

        categorical = self._infer_categorical(X, feature_cols)
        continuous = [c for c in feature_cols if c not in categorical]

        self.feature_names_ = list(feature_cols)
        self.categorical_features_ = categorical
        self.continuous_features_ = continuous

        monotone = self.monotone or {}

        feature_configs = {}
        for col in feature_cols:
            if col in categorical:
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                # Pin NumPy output so a global set_config(transform_output=
                # "pandas") elsewhere in the session can't make transform return
                # a DataFrame (which lacks .ravel()).
                encoder.set_output(transform="default")
                # Fit on a bare NumPy column so later transforms of NumPy grids
                # (in effect()/plotting) don't trip sklearn's feature-name check.
                encoder.fit(X[[col]].to_numpy(dtype=object))
                n_cat = len(encoder.categories_[0])
                self.encoders_[col] = encoder
                self.cardinalities_[col] = n_cat
                # Reserve one extra slot for unseen categories at predict time.
                feature_configs[col] = {
                    "type": "categorical",
                    "num_categories": n_cat + 1,
                }
            else:
                scaler = StandardScaler()
                # Pin NumPy output (see OrdinalEncoder note above).
                scaler.set_output(transform="default")
                col_arr = X[[col]].to_numpy(dtype=float)
                scaler.fit(col_arr)
                self.scalers_[col] = scaler
                std_vals = scaler.transform(col_arr).ravel()
                cfg = {
                    "type": "continuous",
                    "bounds": (float(std_vals.min()), float(std_vals.max())),
                }
                if col in monotone:
                    cfg["monotonicity"] = monotone[col]
                feature_configs[col] = cfg

        self.feature_configs_ = feature_configs

    def _build(self, X, y) -> None:
        """Infer types/bounds, fit preprocessing, and instantiate the module.

        The non-selection code path (``features`` is ``"all"`` or a list and
        ``interactions`` is ``[]``, ``"all"`` or a list). Selection paths build
        the module directly in :meth:`fit` after resolving the terms.
        """
        feature_cols = self._resolve_feature_columns(X, y)
        self._fit_preprocessing(X, y, feature_cols)

        # Resolve interactions ([], "all" or an explicit list).
        self.interactions_ = self._resolve_interactions(feature_cols)
        self.features_ = list(feature_cols)

        self.module_ = self._make_module()

    def _make_module(self) -> ANAMModule:
        """
        Instantiate the raw engine module from the resolved fitted state.

        Uses ``self.feature_configs_``, ``self.interactions_`` and ``self.link_``
        together with the architecture hyperparameters. Shared by :meth:`_build`
        (fresh construction) and :meth:`load` (reconstruction before restoring a
        ``state_dict``) so the two paths cannot drift.
        """
        return ANAMModule(
            feature_configs=self.feature_configs_,
            interactions=self.interactions_,
            num_layers_main=self.num_layers,
            num_neurons_main=self.num_neurons,
            num_layers_interaction=(
                self.num_layers_interaction
                if self.num_layers_interaction is not None
                else self.num_layers
            ),
            num_neurons_interaction=(
                self.num_neurons_interaction
                if self.num_neurons_interaction is not None
                else self.num_neurons
            ),
            num_keypoints=self.num_keypoints,
            num_vertices=self.num_vertices,
            link=self.link_,
        )

    def _resolve_interactions(self, feature_cols: list[str]) -> list[tuple[str, str]]:
        """Resolve the ``interactions`` argument to a concrete list of pairs."""
        if isinstance(self.interactions, str):
            if self.interactions == "all":
                return list(combinations(feature_cols, 2))
            if self.interactions == "auto":
                # Unreachable on the normal path: fit() resolves "auto" through
                # its selection pipeline (rank_interactions -> choose) and never
                # calls _build/_resolve_interactions for it. Only reachable by
                # driving the private build path directly.
                raise RuntimeError(
                    "interactions='auto' is resolved by fit(); call fit(...) "
                    "instead of building the model directly."
                )
            raise ValueError(
                f"Unknown interactions={self.interactions!r}; use 'auto', 'all', "
                "[], or a list of pairs."
            )
        return [tuple(pair) for pair in self.interactions]

    def _encode(self, X, columns: list[str] | None = None) -> dict[str, Tensor]:
        """Transform a raw DataFrame into the module's input dict of tensors.

        Encodes ``self.feature_names_`` by default, or an explicit ``columns``
        subset/superset (used during screening to encode the full candidate
        pool before the final feature set is known).
        """
        columns = columns if columns is not None else self.feature_names_
        if columns is None:
            raise RuntimeError("Model is not built yet; call fit() first.")

        x: dict[str, Tensor] = {}
        for col in columns:
            if col in self.categorical_features_:
                encoder = self.encoders_[col]
                codes = encoder.transform(X[[col]].to_numpy(dtype=object)).ravel()
                # Unseen categories (-1) map to the reserved final slot.
                n_cat = self.cardinalities_[col]
                codes = np.where(codes < 0, n_cat, codes)
                x[col] = torch.tensor(codes, dtype=torch.long)
            else:
                scaler = self.scalers_[col]
                vals = scaler.transform(X[[col]].to_numpy(dtype=float)).ravel()
                x[col] = torch.tensor(vals, dtype=torch.float32)
        return x

    # ------------------------------------------------------------------
    # Offset / target helpers
    # ------------------------------------------------------------------
    def _resolve_offset(self, X, offset=None) -> Tensor | None:
        """Resolve the offset tensor from an explicit array or exposure column."""
        if offset is not None:
            arr = np.asarray(offset, dtype=float).ravel()
        elif self.exposure is not None and self.exposure in X.columns:
            arr = np.asarray(X[self.exposure].values, dtype=float).ravel()
            if self.link_ == "exp":
                arr = np.log(arr)
        else:
            return None
        return torch.tensor(arr, dtype=torch.float32)

    @staticmethod
    def _as_target_tensor(y) -> Tensor:
        arr = np.asarray(getattr(y, "values", y), dtype=float).ravel()
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(-1)

    # ------------------------------------------------------------------
    # Variable selection
    # ------------------------------------------------------------------
    def _screening_config(
        self, ensemble_size, ensemble_epochs, ensemble_patience, verbose
    ):
        """Assemble a :class:`~anam.sparsity.SparsityConfig` for the screens.

        Subnet sizes come from the wrapper's architecture; ``ensemble_size`` and
        the optional epoch/patience overrides come from the ranking methods. When
        an
        epoch/patience override is given it is applied to *all three* screening
        stages (ensemble, baseline, interaction) so a fast screen stays fast end
        to end; otherwise the ``SparsityConfig`` defaults stand.
        """
        cfg = SparsityConfig(
            num_ensemble=ensemble_size,
            ensemble_num_layers=self.num_layers,
            ensemble_num_neurons=self.num_neurons,
            interaction_num_layers=(
                self.num_layers_interaction
                if self.num_layers_interaction is not None
                else self.num_layers
            ),
            interaction_num_neurons=(
                self.num_neurons_interaction
                if self.num_neurons_interaction is not None
                else self.num_neurons
            ),
            # A moderate rate keeps the many short screening runs converging.
            learning_rate=1e-2,
            verbose=verbose,
        )
        if ensemble_epochs is not None:
            cfg.ensemble_epochs = ensemble_epochs
            cfg.baseline_epochs = ensemble_epochs
            cfg.interaction_epochs = ensemble_epochs
        if ensemble_patience is not None:
            cfg.ensemble_patience = ensemble_patience
            cfg.baseline_patience = ensemble_patience
            cfg.interaction_patience = ensemble_patience
        return cfg

    def _stage1_fingerprint(self, candidate_cols, n_rows, cfg, base_seed, has_offset):
        """Cheap fingerprint of everything stage 1 depends on."""
        return (
            tuple(candidate_cols),
            int(n_rows),
            cfg.num_ensemble,
            cfg.ensemble_epochs,
            cfg.ensemble_patience,
            cfg.ensemble_num_layers,
            cfg.ensemble_num_neurons,
            cfg.learning_rate,
            cfg.batch_size,
            int(base_seed),
            self.distribution,
            self.link_,
            bool(has_offset),
        )

    def _screen_main_effects(
        self,
        X,
        y,
        X_val,
        y_val,
        ensemble_size,
        ensemble_epochs,
        ensemble_patience,
        validation_split,
        random_state,
        verbose,
    ) -> dict:
        """Stage-1 screening shared by the ranking methods and ``fit``.

        Resolves the validation set, fits preprocessing on the full candidate
        pool, trains the main-effects ensemble (cached on ``_stage1_cache``) and
        stores the sorted scores in ``main_importance_``. Returns the encoded
        artifacts Stage 2 needs so :meth:`rank_interactions` can continue without
        retraining the ensemble.
        """
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        base_seed = int(random_state) if random_state is not None else 0

        # Resolve a validation set.
        if X_val is None:
            if not (validation_split and validation_split > 0):
                raise ValueError(
                    "Variable selection needs validation data: pass X_val/y_val "
                    "or use validation_split > 0 (selection is scored on a "
                    "held-out set)."
                )
            idx = np.arange(len(X))
            train_idx, val_idx = train_test_split(
                idx, test_size=validation_split, random_state=random_state
            )
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
        else:
            X_train, y_train = X, y

        # Fit preprocessing on the candidate pool (once, unless columns change).
        candidate_cols = self._candidate_columns(X_train, y_train)
        if (
            self._candidate_feature_configs_ is None
            or set(self._candidate_feature_configs_) != set(candidate_cols)
        ):
            self._fit_preprocessing(X_train, y_train, candidate_cols)
            self._candidate_feature_configs_ = dict(self.feature_configs_)

        # Encode train/val once over the whole candidate pool.
        train_x = self._encode(X_train, candidate_cols)
        val_x = self._encode(X_val, candidate_cols)
        train_y = self._as_target_tensor(y_train)
        val_y = self._as_target_tensor(y_val)
        train_offset = self._resolve_offset(X_train)
        val_offset = self._resolve_offset(X_val)

        cfg = self._screening_config(
            ensemble_size, ensemble_epochs, ensemble_patience, verbose
        )
        has_offset = train_offset is not None

        # ---- Stage 1: main-effect ensemble ranking (cached) ----
        fp1 = self._stage1_fingerprint(
            candidate_cols, len(train_y), cfg, base_seed, has_offset
        )
        if self._stage1_cache is None or self._stage1_cache["key"] != fp1:
            ensemble = sparsity.train_ensemble_main_effects(
                self._candidate_feature_configs_,
                train_x,
                train_y,
                val_x,
                val_y,
                loss_fn=self.loss_fn_,
                config=cfg,
                train_offset=train_offset,
                val_offset=val_offset,
                base_seed=base_seed,
            )
            importance = sparsity.compute_ensemble_importance(ensemble)
            self._stage1_cache = {"key": fp1, "importance": importance}
        importance = self._stage1_cache["importance"]

        self.main_importance_ = pd.Series(importance).sort_values(ascending=False)

        return {
            "candidate_cols": candidate_cols,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "train_x": train_x,
            "val_x": val_x,
            "train_y": train_y,
            "val_y": val_y,
            "train_offset": train_offset,
            "val_offset": val_offset,
            "cfg": cfg,
            "base_seed": base_seed,
            "fp1": fp1,
        }

    def rank_main_effects(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        ensemble_size: int = 10,
        ensemble_epochs: int | None = None,
        ensemble_patience: int | None = None,
        validation_split: float = 0.2,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> "ANAM":
        """
        Rank every candidate main effect by ensemble subnet variance (Stage 1).

        This is the first step of the paper's manual workflow: it populates
        ``main_importance_`` for *all* candidate columns — without choosing a
        cutoff — so you can inspect the scree plot via
        :meth:`plot_main_effect_importance`, decide how many mains to keep, then
        rank the interactions those mains imply with :meth:`rank_interactions`.
        The ensemble is cached, so a following :meth:`rank_interactions` (or a
        ``fit`` under ``"auto"`` selection) on the same data and screening
        settings reuses it rather than retraining.

        Calling this is optional: :meth:`rank_interactions` and ``fit`` (under
        ``"auto"`` selection) run it automatically. Scored on validation data:
        pass ``X_val``/``y_val`` explicitly, or a seeded ``validation_split``
        slice of ``(X, y)`` is held out (``validation_split=0`` with no ``X_val``
        raises). Returns ``self``.
        """
        self._screen_main_effects(
            X,
            y,
            X_val,
            y_val,
            ensemble_size,
            ensemble_epochs,
            ensemble_patience,
            validation_split,
            random_state,
            verbose,
        )
        return self

    def _resolve_k_main(self, n_main) -> int:
        """Resolve ``n_main`` (``"auto"`` elbow or an int) against the ranking.

        Requires :meth:`rank_main_effects` to have populated ``main_importance_``.
        """
        if self.main_importance_ is None:
            raise RuntimeError(
                "Main effects have not been ranked yet; call rank_main_effects() "
                "(or fit() under 'auto' selection) first."
            )
        if isinstance(n_main, str):
            if n_main != "auto":
                raise ValueError(f"Unknown n_main={n_main!r}; use 'auto' or an int.")
            return sparsity._find_elbow(self.main_importance_.values)
        return max(1, min(int(n_main), len(self.main_importance_)))

    def rank_interactions(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        n_main="auto",
        ensemble_size: int = 10,
        ensemble_epochs: int | None = None,
        ensemble_patience: int | None = None,
        validation_split: float = 0.2,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> "ANAM":
        """
        Rank the interactions implied by the top main effects (Stage 2).

        Under strong heredity a pairwise term ``(a, b)`` is a candidate only when
        both ``a`` and ``b`` are kept main effects, so this step needs to know how
        many mains you are keeping: ``n_main`` (``"auto"`` elbow, or an int) picks
        the top mains from the Stage-1 ranking, and every within-heredity pair of
        those is scored by how much it improves the validation loss over a
        mains-only baseline. Populates ``interaction_scores_`` (a descending
        Series of Δ validation loss over the pairs that beat the baseline),
        ``baseline_val_loss_`` and the internal candidate losses backing
        :meth:`plot_interaction_importance`.

        Stage 1 is run first if needed (and cached), so calling
        :meth:`rank_main_effects` beforehand is optional — but doing so lets you
        inspect the mains and choose ``n_main`` deliberately. This ranks; it does
        not commit — call :meth:`choose` (or ``fit``) to fix the final terms.

        Scored on validation data (see :meth:`rank_main_effects` for the
        ``X_val``/``validation_split`` rules). Returns ``self``.
        """
        # Stage 1: rank every candidate main effect (ensemble cached, so a prior
        # rank_main_effects() on the same data/settings is reused here).
        screen = self._screen_main_effects(
            X,
            y,
            X_val,
            y_val,
            ensemble_size,
            ensemble_epochs,
            ensemble_patience,
            validation_split,
            random_state,
            verbose,
        )
        train_x, val_x = screen["train_x"], screen["val_x"]
        train_y, val_y = screen["train_y"], screen["val_y"]
        train_offset, val_offset = screen["train_offset"], screen["val_offset"]
        cfg = screen["cfg"]
        base_seed = screen["base_seed"]
        fp1 = screen["fp1"]

        # The mains whose within-heredity pairs we screen over.
        k_main = self._resolve_k_main(n_main)
        selected_mains = list(self.main_importance_.index[:k_main])
        self._interaction_basis_ = selected_mains

        # ---- Stage 2: interaction screen on the selected mains (cached) ----
        fp2 = (
            fp1,
            frozenset(selected_mains),
            cfg.baseline_epochs,
            cfg.baseline_patience,
            cfg.interaction_epochs,
            cfg.interaction_patience,
            cfg.interaction_num_layers,
            cfg.interaction_num_neurons,
        )
        if self._stage2_cache is None or self._stage2_cache["key"] != fp2:
            selected_configs = {
                k: self._candidate_feature_configs_[k] for k in selected_mains
            }
            sel_train_x = {k: train_x[k] for k in selected_mains}
            sel_val_x = {k: val_x[k] for k in selected_mains}

            baseline_model, baseline_loss = sparsity.train_main_effects_model(
                selected_configs,
                sel_train_x,
                train_y,
                sel_val_x,
                val_y,
                loss_fn=self.loss_fn_,
                config=cfg,
                seed=base_seed,
                train_offset=train_offset,
                val_offset=val_offset,
            )
            candidates = list(combinations(selected_mains, 2))
            inter_losses = sparsity.evaluate_candidate_interactions(
                baseline_model,
                selected_configs,
                candidates,
                sel_train_x,
                train_y,
                sel_val_x,
                val_y,
                loss_fn=self.loss_fn_,
                config=cfg,
                baseline_val_loss=baseline_loss,
                train_offset=train_offset,
                val_offset=val_offset,
            )
            self._stage2_cache = {
                "key": fp2,
                "baseline_val_loss": baseline_loss,
                "interaction_losses": inter_losses,
            }
        baseline_loss = self._stage2_cache["baseline_val_loss"]
        inter_losses = self._stage2_cache["interaction_losses"]
        self.baseline_val_loss_ = baseline_loss
        self._interaction_losses_ = dict(inter_losses)

        # Δ val-loss over the baseline; keep only pairs that beat it.
        beating = {
            pair: baseline_loss - loss
            for pair, loss in inter_losses.items()
            if (baseline_loss - loss) > 0
        }
        beating_sorted = sorted(beating.items(), key=lambda kv: kv[1], reverse=True)
        self.interaction_scores_ = pd.Series(
            [d for _, d in beating_sorted],
            index=pd.Index([p for p, _ in beating_sorted], tupleize_cols=False),
            dtype=float,
        )
        return self

    def choose(self, n_main="auto", n_interactions="auto") -> "ANAM":
        """
        Commit the final terms from the cached rankings — cheap, no data needed.

        This is the look-then-decide payoff: after inspecting
        :meth:`plot_main_effect_importance` and :meth:`plot_interaction_importance`
        you fix how many of each to keep. ``n_main``/``n_interactions`` accept
        ``"auto"`` (the elbow rule on the sorted scores) or an explicit int. It
        only slices rankings already computed by :meth:`rank_main_effects` /
        :meth:`rank_interactions` — no training happens here — and sets
        ``features_`` (top ``n_main`` mains) and ``interactions_`` (top
        ``n_interactions`` pairs), which ``fit`` then respects.

        Raises if the rankings it needs are missing, or if ``n_main`` disagrees
        with the mains :meth:`rank_interactions` ranked its pairs against (strong
        heredity would otherwise be violated) — re-run ``rank_interactions`` with
        the matching ``n_main``. Returns ``self``.
        """
        k_main = self._resolve_k_main(n_main)
        selected_mains = list(self.main_importance_.index[:k_main])
        self.features_ = selected_mains

        if isinstance(n_interactions, str):
            if n_interactions != "auto":
                raise ValueError(
                    f"Unknown n_interactions={n_interactions!r}; use 'auto' or an int."
                )
            want_any = True
            auto_int = True
        else:
            n_interactions = int(n_interactions)
            want_any = n_interactions > 0
            auto_int = False

        if not want_any:
            self.interactions_ = []
            self._selection_ran = True
            return self

        if self.interaction_scores_ is None or self._interaction_basis_ is None:
            raise RuntimeError(
                "Interactions have not been ranked yet; call rank_interactions() "
                "first (or pass n_interactions=0 for a pure main-effects model)."
            )
        if set(self._interaction_basis_) != set(selected_mains):
            raise ValueError(
                f"rank_interactions() ranked pairs over the top "
                f"{len(self._interaction_basis_)} main effects, but "
                f"choose(n_main={k_main}) selects a different set of mains; "
                f"re-run rank_interactions(..., n_main={k_main}) so the "
                "candidate pairs respect strong heredity."
            )

        if auto_int:
            k_int = (
                sparsity._find_elbow(self.interaction_scores_.values)
                if len(self.interaction_scores_) > 0
                else 0
            )
        else:
            k_int = min(n_interactions, len(self.interaction_scores_))
        self.interactions_ = [
            tuple(p) for p in list(self.interaction_scores_.index[:k_int])
        ]

        self._selection_ran = True
        return self

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def _resolve_final_features(self, candidate_cols: list[str]) -> list[str]:
        """Resolve the constructor ``features`` value to a concrete list.

        ``"auto"`` defers to the mains chosen by :meth:`choose` (already stored
        in ``self.features_``); ``"all"`` is the full candidate pool; a list is
        honoured (dropping any exposure/target column).
        """
        if self.features == "auto":
            return list(self.features_)
        if self.features == "all":
            return list(candidate_cols)
        if isinstance(self.features, str):
            raise ValueError(
                f"Unknown features={self.features!r}; use 'auto', 'all', or a list."
            )
        return [c for c in self.features if c in candidate_cols]

    def _resolve_final_interactions(
        self, final_features: list[str]
    ) -> list[tuple[str, str]]:
        """Resolve the constructor ``interactions`` value to concrete pairs."""
        if self.interactions == "auto":
            feats = set(final_features)
            return [
                (a, b)
                for (a, b) in (self.interactions_ or [])
                if a in feats and b in feats
            ]
        if self.interactions == "all":
            if len(final_features) > 10:
                warnings.warn(
                    f"interactions='all' with {len(final_features)} features builds "
                    f"{len(final_features) * (len(final_features) - 1) // 2} "
                    "interaction subnets; consider 'auto' or an explicit list.",
                    stacklevel=2,
                )
            return list(combinations(final_features, 2))
        if isinstance(self.interactions, str):
            raise ValueError(
                f"Unknown interactions={self.interactions!r}; use 'auto', 'all', "
                "[], or a list of pairs."
            )
        return [tuple(pair) for pair in self.interactions]

    def plot_main_effect_importance(
        self, n_main=None, ax=None, show_elbow: bool = True
    ):
        """
        Bar chart of the Stage-1 main-effect importances over all candidates.

        This is the plot for the manual workflow: run :meth:`rank_main_effects`
        (or ``fit``), then inspect the scree here before deciding how many mains
        to keep. Pass ``n_main`` to draw the cutoff you are considering; with the
        default ``None`` and ``show_elbow`` the automatic elbow cutoff is marked
        instead, so you can see where ``"auto"`` would draw the boundary before
        overriding it. Returns the matplotlib axis.
        """
        if self.main_importance_ is None:
            raise RuntimeError(
                "No importances yet; call rank_main_effects() (or fit()) first."
            )
        if n_main is not None:
            cutoff = max(1, min(int(n_main), len(self.main_importance_)))
        elif show_elbow:
            cutoff = sparsity._find_elbow(self.main_importance_.values)
        else:
            cutoff = None
        return sparsity.plot_main_effect_importance(
            self.main_importance_.to_dict(),
            ax=ax,
            title="Subnet Variances by Covariate",
            cutoff=cutoff,
        )

    def plot_interaction_importance(
        self, n_interactions=None, ax=None, show_elbow: bool = True
    ):
        """
        Bar chart of the Stage-2 candidate-interaction improvements.

        Run :meth:`rank_interactions` first; this shows each candidate pair's
        validation-loss improvement over the mains-only baseline (descending),
        which is what selection ranks on. Pass ``n_interactions`` to draw the
        cutoff you are considering; with the default ``None`` and ``show_elbow``
        the automatic elbow cutoff is marked instead. Returns the matplotlib
        axis.
        """
        if self._interaction_losses_ is None or self.baseline_val_loss_ is None:
            raise RuntimeError(
                "No interaction ranking yet; call rank_interactions() first."
            )
        if n_interactions is not None:
            cutoff = int(n_interactions) if n_interactions > 0 else None
        elif show_elbow and self.interaction_scores_ is not None and len(
            self.interaction_scores_
        ) > 0:
            cutoff = sparsity._find_elbow(self.interaction_scores_.values)
        else:
            cutoff = None
        return sparsity.plot_interaction_selection(
            self._interaction_losses_,
            self.baseline_val_loss_,
            ax=ax,
            cutoff=cutoff,
        )

    def plot_selection(self, figsize=(14, 5)):
        """
        Plot the two screening stages with the chosen cutoffs marked.

        Left panel: main-effect ensemble importances (bars, descending) with a
        dashed vertical line after the last selected feature. Right panel: the
        candidate-interaction validation losses against the baseline, with a
        dashed line after the last selected interaction. Requires that ranking
        and :meth:`choose` have run (or a ``fit`` that triggered ``"auto"``
        selection). Returns the figure.
        """
        if self.main_importance_ is None:
            raise RuntimeError(
                "No selection to plot; call rank_main_effects()/rank_interactions() "
                "(then choose()), or fit() with features/interactions='auto' first."
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        sparsity.plot_main_effect_importance(self.main_importance_.to_dict(), ax=ax1)
        n_main = len(self.features_) if self.features_ is not None else 0
        if 0 < n_main < len(self.main_importance_):
            ax1.axvline(n_main - 0.5, color="red", linestyle="--",
                        label=f"cutoff (top {n_main})")
            ax1.legend()

        losses = self._interaction_losses_ or {}
        if losses:
            sparsity.plot_interaction_selection(
                losses, self.baseline_val_loss_, ax=ax2
            )
            n_int = len(self.interactions_) if self.interactions_ is not None else 0
            if n_int > 0:
                ax2.axvline(n_int - 0.5, color="green", linestyle="--",
                            label=f"cutoff (top {n_int})")
                ax2.legend()
        else:
            ax2.set_title("No candidate interactions")

        fig.tight_layout()
        # Detach from pyplot so notebook/Quarto inline backends don't auto-show
        # this figure *in addition* to the returned object (which would render
        # it twice). savefig() and the returned handle still work after close.
        plt.close(fig)
        return fig

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs: int = 300,
        batch_size: int = 5000,
        lr: float = 1e-2,
        patience: int = 30,
        offset=None,
        offset_val=None,
        validation_split: float = 0.2,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> "ANAM":
        """
        Fit the model on raw DataFrames.

        If no explicit validation set is supplied and ``validation_split > 0``, a
        seeded slice of the training data is held out for early stopping (and,
        under ``"auto"`` selection, for scoring). ``validation_split=0`` trains on
        everything with no early stopping.

        Returns ``self``; the training curves are stored in ``self.history_``.
        """
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Carve out a validation split if none was provided.
        if X_val is None and validation_split and validation_split > 0:
            idx = np.arange(len(X))
            train_idx, val_idx = train_test_split(
                idx, test_size=validation_split, random_state=random_state
            )
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
        else:
            X_train, y_train = X, y

        # Resolve the feature set / interactions, running selection if the
        # constructor asked for "auto" and it has not been resolved yet.
        needs_selection = self.features == "auto" or self.interactions == "auto"
        if needs_selection and not self._selection_ran:
            if X_val is None:
                raise ValueError(
                    "'auto' feature/interaction selection needs validation data: "
                    "pass X_val/y_val or use validation_split > 0 "
                    "(validation_split=0 with 'auto' is not allowed)."
                )
            # Fully-automatic path: rank both stages and commit with the elbow
            # rule. A prior manual choose() sets _selection_ran and is respected
            # here (this branch is skipped). Interaction ranking is only needed
            # when interactions are themselves "auto".
            self.rank_main_effects(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                validation_split=0.0,
                random_state=random_state,
                verbose=verbose,
            )
            if self.interactions == "auto":
                self.rank_interactions(
                    X_train,
                    y_train,
                    X_val=X_val,
                    y_val=y_val,
                    n_main="auto",
                    validation_split=0.0,
                    random_state=random_state,
                    verbose=verbose,
                )
                self.choose(n_main="auto", n_interactions="auto")
            else:
                # Interactions come from the constructor ([], "all" or a list);
                # only the mains need choosing here.
                self.choose(n_main="auto", n_interactions=0)

        if self._selection_ran:
            # Selection fitted preprocessing on the candidate pool; build the
            # final module on exactly the resolved terms.
            candidate_cols = list(self._candidate_feature_configs_.keys())
            final_features = self._resolve_final_features(candidate_cols)
            final_interactions = self._resolve_final_interactions(final_features)

            self.feature_configs_ = {
                k: self._candidate_feature_configs_[k] for k in final_features
            }
            self.feature_names_ = final_features
            self.categorical_features_ = [
                c for c in final_features if c in self.categorical_features_
            ]
            self.continuous_features_ = [
                c for c in final_features if c in self.continuous_features_
            ]
            self.features_ = final_features
            self.interactions_ = final_interactions
            self.module_ = self._make_module()
        elif self.module_ is None:
            # Lazy build on first fit (preprocessing fitted on the training slice).
            self._build(X_train, y_train)

        train_x = self._encode(X_train)
        train_y = self._as_target_tensor(y_train)
        train_offset = self._resolve_offset(X_train, offset)

        if X_val is not None:
            val_x = self._encode(X_val)
            val_y = self._as_target_tensor(y_val)
            val_offset = self._resolve_offset(X_val, offset_val)
        else:
            val_x = val_y = val_offset = None

        smooth_features = (
            list(self.continuous_features_) if self.smoothness > 0 else None
        )
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            lambda_smooth=self.smoothness,
            lambda_mc=self.marginal_clarity,
            smooth_features=smooth_features,
            patience=patience,
            verbose=verbose,
        )

        self.history_ = train_anam(
            self.module_,
            train_x,
            train_y,
            val_x,
            val_y,
            train_offset=train_offset,
            val_offset=val_offset,
            config=config,
            loss_fn=self.loss_fn_,
        )

        # Cache per-term importances (variance of each shape function over the
        # training data) so importance()/plot_importance() need no recomputation.
        importances = self.module_.get_variable_importance(train_x)
        self.term_importances_ = pd.Series(importances).sort_values(ascending=False)

        # Cache deciles of the training data for plot_effect()'s data-density
        # ticks (mirrors scikit-learn's PartialDependenceDisplay decile markers).
        self.deciles_ = {
            col: np.percentile(
                np.asarray(X_train[col]), np.arange(10, 100, 10)
            )
            for col in self.continuous_features_
        }
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict the response-scale mean ``E[Y]`` for raw DataFrame ``X``.

        The exposure offset (if configured) is read from ``X`` and included.
        Returns a flat NumPy array.
        """
        if self.module_ is None:
            raise RuntimeError("Model is not fitted yet; call fit() first.")

        x = self._encode(X)
        offset = self._resolve_offset(X)

        device = next(self.module_.parameters()).device
        x = {k: v.to(device) for k, v in x.items()}
        if offset is not None:
            offset = offset.to(device)

        self.module_.eval()
        with torch.no_grad():
            pred = self.module_(x, offset=offset)
        return pred.squeeze(-1).cpu().numpy()

    def predict_components(self, X):
        """
        Decompose the linear predictor into per-term contributions.

        Returns a :class:`pandas.DataFrame` with one column per fitted term (main
        effect names and interaction names formatted ``"a_b"``), plus a ``bias``
        column and an ``offset`` column, all on the LINK scale (the additive
        contributions *before* the link is applied). Summing the columns row-wise
        reproduces the linear predictor, so for a log link
        ``predict_components(X).sum(axis=1)`` equals ``log(predict(X))``.
        """
        if self.module_ is None:
            raise RuntimeError("Model is not fitted yet; call fit() first.")

        x = self._encode(X)
        offset = self._resolve_offset(X)

        device = next(self.module_.parameters()).device
        x = {k: v.to(device) for k, v in x.items()}
        offset_t = offset.to(device) if offset is not None else None

        self.module_.eval()
        with torch.no_grad():
            _, mains, inters = self.module_(
                x, offset=offset_t, return_components=True
            )

        n_rows = len(next(iter(x.values())))
        data: dict[str, np.ndarray] = {}
        for name in self.feature_names_:
            data[name] = mains[name].squeeze(-1).cpu().numpy()
        for (a, b), out in inters.items():
            data[f"{a}_{b}"] = out.squeeze(-1).cpu().numpy()
        data["bias"] = np.full(n_rows, float(self.module_.bias.item()))
        if offset is not None:
            data["offset"] = offset.cpu().numpy().ravel()
        else:
            data["offset"] = np.zeros(n_rows)

        index = getattr(X, "index", None)
        return pd.DataFrame(data, index=index)

    # ------------------------------------------------------------------
    # Interpretation (original units)
    # ------------------------------------------------------------------
    def _axis_grid(self, name: str, n_points: int = 100, for_interaction: bool = False):
        """
        Build an evaluation grid for one feature.

        Returns ``(grid_std, axis, is_categorical)`` where ``grid_std`` is the
        torch tensor on the STANDARDISED/coded scale to feed the engine, and
        ``axis`` is the matching ORIGINAL-units array (continuous values or
        category labels) for plotting.
        """
        if name in self.categorical_features_:
            n_cat = self.cardinalities_[name]
            # Interaction MLPs consume codes as floats; embeddings need longs.
            dtype = torch.float32 if for_interaction else torch.long
            grid_std = torch.arange(n_cat, dtype=dtype)
            labels = np.asarray(self.encoders_[name].categories_[0])
            return grid_std, labels, True

        lo, hi = self.feature_configs_[name]["bounds"]
        grid_std = torch.linspace(float(lo), float(hi), n_points)
        grid_orig = (
            self.scalers_[name]
            .inverse_transform(grid_std.numpy().reshape(-1, 1))
            .ravel()
        )
        return grid_std, grid_orig, False

    def effect(self, name: str, other: str | None = None):
        """
        Low-level accessor for a fitted shape function in ORIGINAL units.

        Main effect (``other=None``):
            - continuous: returns ``(grid_original, values)`` on a 100-point grid
              spanning the training column's raw min/max.
            - categorical: returns ``(category_labels, values)`` evaluated at each
              category code, with the original string/category labels.

        Interaction (``effect(a, b)``): returns ``(grid_a, grid_b, values_2d)``
        where ``values_2d[i, j]`` is the interaction at ``grid_a[i]``,
        ``grid_b[j]``. Categorical axes use category codes/labels.
        """
        if self.module_ is None:
            raise RuntimeError("Model is not fitted yet; call fit() first.")

        self.module_.eval()

        if other is None:
            grid_std, axis, _ = self._axis_grid(name)
            with torch.no_grad():
                values = self.module_.get_shape_function(name, grid_std)
            return axis, values.cpu().numpy()

        grid_a, axis_a, _ = self._axis_grid(name, n_points=50, for_interaction=True)
        grid_b, axis_b, _ = self._axis_grid(other, n_points=50, for_interaction=True)
        with torch.no_grad():
            values = self.module_.get_shape_function(name, grid_a, other, grid_b)
        values = values.cpu().numpy()
        # get_shape_function swaps to the stored key order internally; if the
        # subnet is keyed "other_name" the result comes back transposed.
        if f"{name}_{other}" not in self.module_.interaction_subnets:
            values = values.T
        return axis_a, axis_b, values

    def _add_decile_ticks(self, ax, name: str, other: str | None = None):
        """Draw data-density decile ticks on ``ax``, mirroring sklearn's PDP."""
        if self.deciles_ is None:
            return
        if other is None:
            # Main effect: ticks on the x-axis (continuous features only).
            if name not in self.deciles_:
                return
            deciles = self.deciles_[name]
            # Use the axis transform so y=0 is the bottom spine (not the padded
            # data limit); the "|" marker is clipped to the axes, so only the
            # upper half shows, touching the spine.
            ax.plot(deciles, np.zeros_like(deciles), "|",
                    transform=ax.get_xaxis_transform(),
                    color="black", markeredgewidth=1, markersize=8)
        else:
            # Interaction: ticks on both axes (continuous features only).
            for feat, is_x in [(name, True), (other, False)]:
                if feat not in self.deciles_:
                    continue
                deciles = self.deciles_[feat]
                if is_x:
                    ax.plot(deciles, np.zeros_like(deciles), "|",
                            transform=ax.get_xaxis_transform(),
                            color="black", markeredgewidth=1, markersize=8)
                else:
                    ax.plot(np.zeros_like(deciles), deciles, "_",
                            transform=ax.get_yaxis_transform(),
                            color="black", markeredgewidth=1, markersize=8)

    def plot_effect(self, name: str, other: str | None = None, ax=None,
                    show_data_ticks: bool = True):
        """
        Plot a fitted term in original units.

        Continuous main effect → line; categorical main effect → bar chart with
        real category labels; interaction → heatmap. Creates an axis when
        ``ax is None``. Returns the axis.

        When ``show_data_ticks`` is True and the model was fitted with cached
        deciles (``self.deciles_``), small black tick marks are drawn at the
        10th–90th percentiles of the training data along each continuous axis,
        mirroring scikit-learn's ``PartialDependenceDisplay`` so the viewer can
        see where data is dense vs. sparse.
        """
        if other is None:
            if name in self.categorical_features_:
                labels, values = self.effect(name)
                if ax is None:
                    _, ax = plt.subplots(figsize=(6, 4))
                ax.bar([str(lbl) for lbl in labels], values)
            else:
                grid, values = self.effect(name)
                if ax is None:
                    _, ax = plt.subplots(figsize=(6, 4))
                ax.plot(grid, values)
            ax.set_xlabel(name)
            ax.set_ylabel("Partial effect")
            ax.set_title(f"Effect of {name}")
            if show_data_ticks:
                self._add_decile_ticks(ax, name)
            return ax

        # Interaction heatmap.
        axis_a, axis_b, values = self.effect(name, other)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        a_cat = name in self.categorical_features_
        b_cat = other in self.categorical_features_
        a_coords = np.arange(len(axis_a)) if a_cat else np.asarray(axis_a, float)
        b_coords = np.arange(len(axis_b)) if b_cat else np.asarray(axis_b, float)

        # pcolormesh wants C shaped (len(y), len(x)); x = a, y = b → values.T.
        mesh = ax.pcolormesh(
            a_coords, b_coords, values.T, shading="auto", cmap="viridis"
        )
        ax.get_figure().colorbar(mesh, ax=ax)
        if a_cat:
            ax.set_xticks(a_coords)
            ax.set_xticklabels([str(lbl) for lbl in axis_a])
        if b_cat:
            ax.set_yticks(b_coords)
            ax.set_yticklabels([str(lbl) for lbl in axis_b])
        ax.set_xlabel(name)
        ax.set_ylabel(other)
        ax.set_title(f"Interaction: {name} x {other}")
        if show_data_ticks:
            self._add_decile_ticks(ax, name, other)
        return ax

    def plot_effects(self, features=None, ncols: int = 3, figsize=None,
                     show_data_ticks: bool = True):
        """
        Plot a grid of fitted terms in original units.

        With ``features=None`` every fitted term is drawn (main effects in column
        order, then interactions). Otherwise ``features`` is a list of terms:
        strings for main effects and ``(a, b)`` tuples for interactions. Returns
        the figure.
        """
        if self.module_ is None:
            raise RuntimeError("Model is not fitted yet; call fit() first.")

        if features is None:
            terms = [(name, None) for name in self.feature_names_]
            terms += [(a, b) for (a, b) in self.interactions_]
        else:
            terms = []
            for item in features:
                if isinstance(item, (tuple, list)):
                    terms.append((item[0], item[1]))
                else:
                    terms.append((item, None))

        n_terms = len(terms)
        if n_terms == 0:
            raise ValueError("No terms to plot.")
        ncols = min(ncols, n_terms)
        nrows = int(np.ceil(n_terms / ncols))
        if figsize is None:
            figsize = (4 * ncols, 3.2 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()
        for ax, (a, b) in zip(axes, terms):
            self.plot_effect(a, b, ax=ax, show_data_ticks=show_data_ticks)
        for ax in axes[n_terms:]:
            ax.set_visible(False)
        fig.tight_layout()
        # Detach from pyplot so the inline backend doesn't render this figure a
        # second time on top of the returned object (see plot_selection).
        plt.close(fig)
        return fig

    def importance(self):
        """
        Return per-term importances as a descending :class:`pandas.Series`.

        Importance is the variance of each term's shape function over the training
        data (main effects and interactions, the latter keyed ``"a_b"``). Computed
        once at the end of ``fit`` and cached on ``self.term_importances_``.
        """
        if self.term_importances_ is None:
            raise RuntimeError("Model is not fitted yet; call fit() first.")
        return self.term_importances_

    def plot_importance(self, ax=None):
        """Plot :meth:`importance` as a horizontal bar chart. Returns the axis."""
        imp = self.importance()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        ax.barh(list(imp.index[::-1]), list(imp.values[::-1]))
        ax.set_xlabel("Variance")
        ax.set_title("Variable importance")
        return ax

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    # Constructor arguments stored verbatim; a saved model records these so a
    # fresh wrapper can be reconstructed with ``ANAM(**constructor_kwargs)``.
    _CONSTRUCTOR_KWARGS = (
        "distribution",
        "features",
        "interactions",
        "categorical",
        "monotone",
        "smoothness",
        "marginal_clarity",
        "exposure",
        "num_layers",
        "num_neurons",
        "num_layers_interaction",
        "num_neurons_interaction",
        "num_vertices",
        "num_keypoints",
        "link",
    )

    def save(self, path) -> None:
        """
        Serialise a fitted model to ``path`` as a single ``torch.save`` bundle.

        The bundle contains everything needed to fully reconstruct the fitted
        model without retraining: the package version, all constructor kwargs,
        the fitted preprocessing (scalers/encoders/cardinalities and the derived
        ``feature_configs_``), the resolved feature/interaction/link metadata, the
        engine module's ``state_dict``, and the cached importances and training
        history. The loss function is not pickled (it is a plain function); it is
        re-derived from ``distribution`` on load.

        Raises:
            RuntimeError: if called before :meth:`fit` (nothing to save).
        """
        if self.module_ is None:
            raise RuntimeError("Model is not fitted yet; call fit() before save().")

        constructor_kwargs = {k: getattr(self, k) for k in self._CONSTRUCTOR_KWARGS}

        checkpoint = {
            "format": "anam-checkpoint",
            "version": __import__("anam").__version__,
            "constructor_kwargs": constructor_kwargs,
            # Fitted preprocessing.
            "scalers_": self.scalers_,
            "encoders_": self.encoders_,
            "cardinalities_": self.cardinalities_,
            "feature_configs_": self.feature_configs_,
            # Resolved metadata.
            "feature_names_": self.feature_names_,
            "categorical_features_": self.categorical_features_,
            "continuous_features_": self.continuous_features_,
            "features_": self.features_,
            "interactions_": self.interactions_,
            "link_": self.link_,
            # Engine weights + cached interpretation state.
            "state_dict": self.module_.state_dict(),
            "term_importances_": self.term_importances_,
            "history_": self.history_,
            "deciles_": self.deciles_,
            # Selection diagnostics so plot_selection() works on a reloaded model
            # (the common path when a lecture/notebook caches the fitted model).
            "main_importance_": self.main_importance_,
            "interaction_scores_": self.interaction_scores_,
            "_interaction_losses_": self._interaction_losses_,
            "_interaction_basis_": self._interaction_basis_,
            "baseline_val_loss_": getattr(self, "baseline_val_loss_", None),
            "_selection_ran": self._selection_ran,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path) -> "ANAM":
        """
        Reconstruct a fitted :class:`ANAM` from a bundle written by :meth:`save`.

        Rebuilds the wrapper from the saved constructor kwargs, restores the
        fitted preprocessing and resolved metadata, re-derives the loss function
        from the distribution, rebuilds the engine module and loads its weights,
        and returns a ready-to-predict model (no refit required).

        Raises:
            RuntimeError: if the file is not a v0.2 ANAM checkpoint (e.g. a raw
                ``state_dict`` from a pre-0.2 release), which cannot be loaded and
                must be regenerated with the current :meth:`save`.
        """
        checkpoint = torch.load(path, weights_only=False)

        required = {
            "format",
            "version",
            "constructor_kwargs",
            "feature_configs_",
            "state_dict",
        }
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("format") != "anam-checkpoint"
            or checkpoint.get("version") is None
            or not required.issubset(checkpoint)
        ):
            raise RuntimeError(
                f"{path!r} is not a v0.2 ANAM checkpoint. It looks like a raw "
                "state_dict or an incompatible file; regenerate it with "
                "ANAM.save() using anam >= 0.2."
            )

        model = cls(**checkpoint["constructor_kwargs"])

        # Restore fitted preprocessing and resolved metadata.
        model.scalers_ = checkpoint["scalers_"]
        model.encoders_ = checkpoint["encoders_"]
        model.cardinalities_ = checkpoint["cardinalities_"]
        model.feature_configs_ = checkpoint["feature_configs_"]
        model.feature_names_ = checkpoint["feature_names_"]
        model.categorical_features_ = checkpoint["categorical_features_"]
        model.continuous_features_ = checkpoint["continuous_features_"]
        model.features_ = checkpoint["features_"]
        model.interactions_ = checkpoint["interactions_"]
        model.link_ = checkpoint["link_"]
        model.term_importances_ = checkpoint["term_importances_"]
        model.history_ = checkpoint["history_"]
        model.deciles_ = checkpoint.get("deciles_")
        # Selection diagnostics (absent in pre-0.2.x checkpoints — default safely).
        model.main_importance_ = checkpoint.get("main_importance_")
        model.interaction_scores_ = checkpoint.get("interaction_scores_")
        model._interaction_losses_ = checkpoint.get("_interaction_losses_")
        model._interaction_basis_ = checkpoint.get("_interaction_basis_")
        model.baseline_val_loss_ = checkpoint.get("baseline_val_loss_")
        model._selection_ran = checkpoint.get("_selection_ran", False)

        # Re-derive the loss (functions are not pickled) from the distribution.
        loss_fn, _ = resolve_distribution(model.distribution)
        model.loss_fn_ = loss_fn

        # Rebuild the engine module and restore its weights.
        module = model._make_module()
        module.load_state_dict(checkpoint["state_dict"])
        module.eval()
        model.module_ = module
        return model
