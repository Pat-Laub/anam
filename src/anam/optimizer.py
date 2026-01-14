"""
Training utilities for ANAM.

Includes training loops with support for:
- Projected gradient descent (for monotonicity)
- Smoothness and marginal clarity penalties
- Early stopping
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable
from dataclasses import dataclass

from anam.model import ANAM
from anam.losses import gamma_nll, smoothness_penalty, marginal_clarity_penalty


@dataclass
class TrainingConfig:
    """Configuration for ANAM training."""
    epochs: int = 1000
    batch_size: int = 1000
    learning_rate: float = 0.001
    lambda_smooth: float = 1e-4
    lambda_mc: float = 1e-4
    smooth_features: list[str] = None
    patience: int = 50
    min_delta: float = 1e-6
    grid_size: int = 1000
    verbose: bool = True


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_state = None

    def __call__(self, val_loss: float, model: ANAM) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore_best(self, model: ANAM):
        """Restore model to best state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def compute_smoothness_penalties(
    model: ANAM,
    features: list[str],
    feature_bounds: dict[str, tuple[float, float]],
    lambda_smooth: float,
    grid_size: int = 1000,
    device: torch.device = None,
) -> Tensor:
    """
    Compute smoothness penalties for specified features.

    Args:
        model: ANAM model
        features: List of feature names to apply smoothness penalty
        feature_bounds: Dict mapping feature name to (min, max)
        lambda_smooth: Regularization weight
        grid_size: Number of grid points for finite differences
        device: Torch device

    Returns:
        Total smoothness penalty
    """
    if not features:
        return torch.tensor(0.0, device=device)

    total_penalty = torch.tensor(0.0, device=device)

    for name in features:
        if name not in feature_bounds:
            continue

        min_val, max_val = feature_bounds[name]
        grid = torch.linspace(min_val, max_val, grid_size, device=device)
        h = (max_val - min_val) / (grid_size - 1)

        # Get shape function values
        shape_values = model.get_shape_function(name, grid)

        # Add smoothness penalty
        total_penalty = total_penalty + smoothness_penalty(shape_values, h, lambda_smooth)

    return total_penalty


def train_step(
    model: ANAM,
    batch_x: dict[str, Tensor],
    batch_y: Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    config: TrainingConfig,
    feature_bounds: dict[str, tuple[float, float]],
    device: torch.device,
) -> float:
    """
    Perform one training step.

    Args:
        model: ANAM model
        batch_x: Input batch dict
        batch_y: Target batch
        optimizer: Optimizer
        loss_fn: Loss function
        config: Training configuration
        feature_bounds: Feature bounds for smoothness
        device: Torch device

    Returns:
        Loss value for this step
    """
    model.train()
    optimizer.zero_grad()

    # Decide whether we need component outputs
    need_mc = bool(model.interactions) and config.lambda_mc > 0

    if need_mc:
        # Single forward pass that also gives us component outputs
        pred, main_outputs, int_outputs = model(batch_x, return_components=True)
    else:
        pred = model(batch_x)
        main_outputs = None
        int_outputs = None

    # Primary loss
    loss = loss_fn(batch_y, pred)

    # Smoothness penalty
    if config.smooth_features and config.lambda_smooth > 0:
        smooth_penalty = compute_smoothness_penalties(
            model, config.smooth_features, feature_bounds,
            config.lambda_smooth, config.grid_size, device
        )
        loss = loss + smooth_penalty

    # Marginal clarity penalty using already-computed component outputs
    if need_mc:
        mc_penalty = marginal_clarity_penalty(
            {k: v.squeeze(-1) for k, v in main_outputs.items()},
            {k: v.squeeze(-1) for k, v in int_outputs.items()},
            config.lambda_mc,
        )
        loss = loss + mc_penalty

    # Backward pass
    loss.backward()
    optimizer.step()

    # Project monotonic parameters
    model.project_monotonicity()

    return loss.item()


def evaluate(
    model: ANAM,
    data_x: dict[str, Tensor],
    data_y: Tensor,
    loss_fn: Callable,
) -> float:
    """
    Evaluate model on data.

    Args:
        model: ANAM model
        data_x: Input dict
        data_y: Targets
        loss_fn: Loss function

    Returns:
        Loss value
    """
    model.eval()
    with torch.no_grad():
        pred = model(data_x)
        loss = loss_fn(data_y, pred)
    return loss.item()


def train_anam(
    model: ANAM,
    train_x: dict[str, Tensor],
    train_y: Tensor,
    val_x: dict[str, Tensor] = None,
    val_y: Tensor = None,
    config: TrainingConfig = None,
    loss_fn: Callable = None,
    device: torch.device = None,
) -> dict:
    """
    Train an ANAM model.

    Args:
        model: ANAM model to train
        train_x: Training inputs dict
        train_y: Training targets
        val_x: Validation inputs dict (optional)
        val_y: Validation targets (optional)
        config: Training configuration
        loss_fn: Loss function (defaults to gamma_nll)
        device: Torch device

    Returns:
        Training history dict with 'train_loss' and 'val_loss' lists
    """
    config = config or TrainingConfig()
    loss_fn = loss_fn or gamma_nll
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model to device
    model = model.to(device)

    # Move data to device
    train_x = {k: v.to(device) for k, v in train_x.items()}
    train_y = train_y.to(device)
    if val_x is not None:
        val_x = {k: v.to(device) for k, v in val_x.items()}
        val_y = val_y.to(device)

    # Get feature bounds for smoothness penalty
    feature_bounds = {}
    for name, cfg in model.feature_configs.items():
        if cfg.get('type', 'continuous') == 'continuous' and 'bounds' in cfg:
            feature_bounds[name] = cfg['bounds']

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Early stopping
    early_stopping = EarlyStopping(config.patience, config.min_delta)

    # Training history
    history = {'train_loss': [], 'val_loss': []}

    # Create data indices for batching
    n_samples = train_y.shape[0]
    feature_names = list(train_x.keys())

    for epoch in range(config.epochs):
        # Shuffle indices
        perm = torch.randperm(n_samples, device=device)

        epoch_loss = 0.0
        n_batches = 0

        # Mini-batch training
        for i in range(0, n_samples, config.batch_size):
            batch_idx = perm[i:i + config.batch_size]

            # Create batch
            batch_x = {k: v[batch_idx] for k, v in train_x.items()}
            batch_y = train_y[batch_idx]

            # Training step
            loss = train_step(
                model, batch_x, batch_y, optimizer, loss_fn,
                config, feature_bounds, device
            )
            epoch_loss += loss
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_train_loss)

        # Validation
        if val_x is not None:
            val_loss = evaluate(model, val_x, val_y, loss_fn)
            history['val_loss'].append(val_loss)

            # Early stopping check
            if early_stopping(val_loss, model):
                if config.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                early_stopping.restore_best(model)
                break

            if config.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
        else:
            if config.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}")

    return history


def prepare_data(
    df,
    feature_names: list[str],
    target: str,
    categorical_features: list[str] = None,
) -> tuple[dict[str, Tensor], Tensor]:
    """
    Prepare data for ANAM training.

    Args:
        df: Pandas DataFrame
        feature_names: List of feature column names
        target: Target column name
        categorical_features: List of categorical feature names

    Returns:
        Tuple of (features dict, targets tensor)
    """
    categorical_features = categorical_features or []

    x = {}
    for name in feature_names:
        values = df[name].values
        if name in categorical_features:
            x[name] = torch.tensor(values, dtype=torch.long)
        else:
            x[name] = torch.tensor(values, dtype=torch.float32)

    y = torch.tensor(df[target].values, dtype=torch.float32).unsqueeze(-1)

    return x, y
