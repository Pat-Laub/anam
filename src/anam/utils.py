"""
Utility functions for ANAM.

Includes:
- Data preprocessing
- Feature configuration helpers
- Visualization utilities
"""

import torch
from torch import Tensor
import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import matplotlib.pyplot as plt


def remove_outliers_iqr(
    df: "pd.DataFrame",
    column: str,
    log_transform: bool = True,
    k: float = 1.5,
) -> "pd.DataFrame":
    """
    Remove outliers using IQR method.

    Args:
        df: Input DataFrame
        column: Column to check for outliers
        log_transform: Whether to apply log transform before IQR calculation
        k: IQR multiplier (default 1.5)

    Returns:
        DataFrame with outliers removed
    """
    values = df[column].values
    if log_transform:
        values = np.log(values + 1e-8)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (values >= lower) & (values <= upper)
    return df[mask].copy()


def get_feature_bounds(
    df: "pd.DataFrame",
    features: list[str],
    categorical_features: list[str] = None,
) -> dict[str, tuple[float, float]]:
    """
    Get min/max bounds for continuous features.

    Args:
        df: Input DataFrame
        features: List of feature names
        categorical_features: List of categorical feature names (excluded)

    Returns:
        Dict mapping feature name to (min, max) tuple
    """
    categorical_features = categorical_features or []
    bounds = {}

    for name in features:
        if name not in categorical_features:
            bounds[name] = (df[name].min(), df[name].max())

    return bounds


def build_feature_configs(
    df: "pd.DataFrame",
    features: list[str],
    categorical_features: list[str] = None,
    monotonic_features: dict[str, str] = None,
) -> dict:
    """
    Build feature configuration dict for ANAM.

    Args:
        df: Input DataFrame
        features: List of feature names
        categorical_features: List of categorical feature names
        monotonic_features: Dict mapping feature name to monotonicity direction

    Returns:
        Feature configuration dict for ANAM constructor
    """
    categorical_features = categorical_features or []
    monotonic_features = monotonic_features or {}

    configs = {}
    for name in features:
        if name in categorical_features:
            configs[name] = {
                'type': 'categorical',
                'num_categories': df[name].nunique(),
            }
        else:
            configs[name] = {
                'type': 'continuous',
                'bounds': (float(df[name].min()), float(df[name].max())),
            }
            if name in monotonic_features:
                configs[name]['monotonicity'] = monotonic_features[name]

    return configs


def train_val_test_split(
    df: "pd.DataFrame",
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = None,
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """
    Split DataFrame into train, validation, and test sets.

    Args:
        df: Input DataFrame
        test_size: Fraction for test set (from total)
        val_size: Fraction for validation set (from remaining after test)
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    # First split: separate test set
    main, test = train_test_split(df, test_size=test_size, random_state=random_state)

    # Second split: separate validation from training
    train, val = train_test_split(main, test_size=val_size, random_state=random_state)

    return train, val, test


def plot_shape_function(
    model,
    feature: str,
    feature_bounds: tuple[float, float],
    ax: "plt.Axes" = None,
    n_points: int = 100,
    title: str = None,
) -> "plt.Axes":
    """
    Plot a main effect shape function.

    Args:
        model: ANAM model
        feature: Feature name
        feature_bounds: (min, max) bounds for the feature
        ax: Matplotlib axes (creates new if None)
        n_points: Number of points for the grid
        title: Plot title

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    min_val, max_val = feature_bounds
    grid = torch.linspace(min_val, max_val, n_points)

    with torch.no_grad():
        values = model.get_shape_function(feature, grid)

    ax.plot(grid.numpy(), values.numpy())
    ax.set_xlabel(feature)
    ax.set_ylabel('Shape Function')
    ax.set_title(title or f'Shape function for {feature}')
    ax.grid(True, alpha=0.3)

    return ax


def plot_interaction_heatmap(
    model,
    feature1: str,
    feature2: str,
    bounds1: tuple[float, float],
    bounds2: tuple[float, float],
    ax: "plt.Axes" = None,
    n_points: int = 50,
    title: str = None,
) -> "plt.Axes":
    """
    Plot an interaction effect as a heatmap.

    Args:
        model: ANAM model
        feature1: First feature name
        feature2: Second feature name
        bounds1: Bounds for first feature
        bounds2: Bounds for second feature
        ax: Matplotlib axes (creates new if None)
        n_points: Number of points per dimension
        title: Plot title

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    grid1 = torch.linspace(bounds1[0], bounds1[1], n_points)
    grid2 = torch.linspace(bounds2[0], bounds2[1], n_points)

    with torch.no_grad():
        values = model.get_shape_function(feature1, grid1, feature2, grid2)

    # Create heatmap
    im = ax.contourf(
        grid1.numpy(), grid2.numpy(), values.numpy().T,
        levels=20, cmap='viridis'
    )
    plt.colorbar(im, ax=ax)

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title(title or f'Interaction: {feature1} x {feature2}')

    return ax


def plot_variable_importance(
    importance: dict[str, float],
    ax: "plt.Axes" = None,
    top_k: int = None,
) -> "plt.Axes":
    """
    Plot variable importance as a bar chart.

    Args:
        importance: Dict mapping variable name to importance score
        ax: Matplotlib axes (creates new if None)
        top_k: Only show top k variables

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    if top_k:
        sorted_items = sorted_items[:top_k]

    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel('Variance')
    ax.set_title('Variable Importance')

    return ax


def generate_synthetic_data(
    n_samples: int = 50000,
    noise_level: float = 1.0,
    random_state: int = None,
) -> "pd.DataFrame":
    """
    Generate synthetic data matching the paper's data generation process.

    Features X1-X10 are uniform(-1, 1).
    Response Y follows Gamma distribution with mean exp(linear predictor).

    Ground truth functions:
        f1(X1) = |X1| * sin(8*X1)
        f2(X2) = 0.5*sin^3(8*X2) - 0.25*cos(4*X2) + 0.25*X2^2
        f34(X3, X4) = -(X3 + 0.5) * exp(-X4)
        f56(X5, X6) = 1.5*sin(2*pi*(X5-0.5)*(X6+0.5))
        f78(X7, X8) = sign(50*(sin(10*X7)+0.5)) * sign(50*(sin(10*X8)-0.5))

    Args:
        n_samples: Number of samples
        noise_level: Dispersion parameter (phi) for Gamma distribution
        random_state: Random seed

    Returns:
        DataFrame with X1-X10 and Y columns
    """
    import pandas as pd

    if random_state is not None:
        np.random.seed(random_state)

    # Generate features
    X = np.random.uniform(-1, 1, (n_samples, 10))

    # Main effects
    f1 = np.abs(X[:, 0]) * np.sin(8 * X[:, 0])
    f2 = 0.5 * np.sin(8 * X[:, 1])**3 - 0.25 * np.cos(4 * X[:, 1]) + 0.25 * X[:, 1]**2

    # Interaction effects
    f34 = -(X[:, 2] + 0.5) * np.exp(-X[:, 3])
    f56 = 1.5 * np.sin(2 * np.pi * (X[:, 4] - 0.5) * (X[:, 5] + 0.5))
    f78 = np.sign(50 * (np.sin(10 * X[:, 6]) + 0.5)) * np.sign(50 * (np.sin(10 * X[:, 7]) - 0.5))

    # Linear predictor
    beta = 6  # bias
    eta = beta + f1 + f2 + f34 + f56 + f78

    # Mean (exponential link)
    mu = np.exp(eta)

    # Generate Gamma response
    # Gamma with mean=mu and dispersion=phi has shape=1/phi and scale=phi*mu
    shape = 1.0 / noise_level
    scale = noise_level * mu
    Y = np.random.gamma(shape, scale)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(1, 11)])
    df['Y'] = Y

    return df
