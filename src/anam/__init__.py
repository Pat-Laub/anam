"""
ANAM - Actuarial Neural Additive Model

A PyTorch implementation of Neural Additive Models with extensions for
actuarial applications including monotonicity constraints, smoothness
penalties, and interaction effects.

Quickstart::

    from anam import ANAM
    model = ANAM(distribution="poisson", exposure="expo",
                 monotone={"bm": "increasing"}, smoothness=1e-3)
    model.fit(train, train["nclaims"], X_val=val, y_val=val["nclaims"])
    model.plot_effects(["ageph", "bm", "power"]); model.predict(val)

The headline export ``ANAM`` is the user-facing wrapper: it owns preprocessing,
maps ``distribution`` to a loss + link, treats ``exposure`` as an offset column,
runs variable selection by default (``features="auto"``/``interactions="auto"``),
and plots shape functions in original units. Power users can build the raw
engine directly via ``ANAMModule`` and the lower-level helpers, all still
exported below.
"""

from anam.model import ANAMModule

# The headline export ``ANAM`` is the user-facing wrapper (api.py). Power users
# who want the raw nn.Module engine can use ``ANAMModule``.
from anam.api import ANAM
from anam.losses import (
    gamma_nll,
    poisson_nll,
    gaussian_nll,
    smoothness_penalty,
    marginal_clarity_penalty,
    DISTRIBUTIONS,
    resolve_distribution,
)
from anam.subnet import Subnet, EmbeddingSubnet
from anam.lattice import PWLCalibration, Lattice, Lattice1D, MonotonicSubnet
from anam.optimizer import train_anam, TrainingConfig, prepare_data
from anam.utils import (
    generate_synthetic_data,
    build_feature_configs,
    remove_outliers_iqr,
    train_val_test_split,
    plot_shape_function,
    plot_interaction_heatmap,
    plot_variable_importance,
)
from anam.sparsity import (
    SparsityConfig,
    SelectionResult,
    train_ensemble_main_effects,
    compute_ensemble_importance,
    select_main_effects,
    train_main_effects_model,
    evaluate_candidate_interactions,
    select_interactions,
    run_variable_selection,
    plot_main_effect_importance,
    plot_interaction_selection,
    _find_elbow,
)

__version__ = "0.2.0"
__all__ = [
    # Model
    "ANAM",
    "ANAMModule",
    # Subnets
    "Subnet",
    "EmbeddingSubnet",
    # Lattice
    "PWLCalibration",
    "Lattice",
    "Lattice1D",
    "MonotonicSubnet",
    # Losses
    "gamma_nll",
    "poisson_nll",
    "gaussian_nll",
    "smoothness_penalty",
    "marginal_clarity_penalty",
    "DISTRIBUTIONS",
    "resolve_distribution",
    # Training
    "train_anam",
    "TrainingConfig",
    "prepare_data",
    # Sparsity / Variable Selection
    "SparsityConfig",
    "SelectionResult",
    "train_ensemble_main_effects",
    "compute_ensemble_importance",
    "select_main_effects",
    "train_main_effects_model",
    "evaluate_candidate_interactions",
    "select_interactions",
    "run_variable_selection",
    "plot_main_effect_importance",
    "plot_interaction_selection",
    # Utilities
    "generate_synthetic_data",
    "build_feature_configs",
    "remove_outliers_iqr",
    "train_val_test_split",
    "plot_shape_function",
    "plot_interaction_heatmap",
    "plot_variable_importance",
]
