"""
ANAM - Actuarial Neural Additive Model

A PyTorch implementation of Neural Additive Models with extensions for
actuarial applications including monotonicity constraints, smoothness
penalties, and interaction effects.
"""

from anam.model import ANAM
from anam.losses import gamma_nll, poisson_nll, smoothness_penalty, marginal_clarity_penalty
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

__version__ = "0.1.0"
__all__ = [
    # Model
    "ANAM",
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
    "smoothness_penalty",
    "marginal_clarity_penalty",
    # Training
    "train_anam",
    "TrainingConfig",
    "prepare_data",
    # Utilities
    "generate_synthetic_data",
    "build_feature_configs",
    "remove_outliers_iqr",
    "train_val_test_split",
    "plot_shape_function",
    "plot_interaction_heatmap",
    "plot_variable_importance",
]
