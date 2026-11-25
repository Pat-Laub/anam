"""
Sparsity constraint implementation for ANAM.

Implements the three-stage variable selection process described in the paper:
1. Main Effect Selection: Train ensemble of NAMs with main effects only
2. Interaction Effect Selection: Add interactions one-by-one to frozen main effects
3. Fine-tuning: Re-train all selected features simultaneously

Reference: Section 3.5 of the ANAM paper.
"""

import copy
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from anam.losses import gamma_nll
from anam.model import ANAM, interaction_supported
from anam.optimizer import TrainingConfig, train_anam


def _find_elbow(scores) -> int:
    """
    Find the elbow ("kneedle") in a descending-sorted score array.

    Returns the number of leading items to keep. The elbow is the point of
    maximum perpendicular distance from the chord joining the first and last
    score — the mechanical version of eyeballing the kink in a scree plot.
    For a high-plateau-then-cliff profile that point sits at the *bottom* of
    the cliff (the first low value), so the count kept is the plateau before
    it: the elbow index itself.

    Guard rails:
    - Fewer than 3 scores → keep all of them (the elbow is ill-defined).
    - Near-collinear / flat profile (no real kink) → keep all.
    - Ties in the distance → keep more rather than fewer.

    Args:
        scores: 1-D array-like of scores, assumed sorted descending.

    Returns:
        Number of leading items to keep (at least 1).
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n < 3:
        return n

    # Coordinates of each point: x = index, y = score.
    x = np.arange(n, dtype=float)
    y = scores

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    # Perpendicular distance of each point from the chord (x0, y0)-(x1, y1).
    dx = x1 - x0
    dy = y1 - y0
    denom = np.hypot(dx, dy)
    if denom == 0:
        return n

    distances = np.abs(dy * (x - x0) - dx * (y - y0)) / denom

    # Near-collinear (flat or perfectly linear) profile: no meaningful elbow.
    score_range = float(y.max() - y.min())
    if distances.max() <= 1e-9 * max(score_range, 1.0):
        return n

    # argmax on the reversed array picks the *last* maximum on ties (keep more),
    # then map the index back to the original orientation.
    rev_idx = int(np.argmax(distances[::-1]))
    elbow_idx = n - 1 - rev_idx

    # The plateau kept is everything strictly above the cliff bottom, i.e. the
    # elbow index itself; guarantee at least one item survives.
    return max(elbow_idx, 1)


@dataclass
class SparsityConfig:
    """Configuration for sparsity-based variable selection.

    Attributes:
        num_ensemble: Number of models in the ensemble for Stage 1.
        ensemble_epochs: Maximum epochs for each ensemble model.
        ensemble_patience: Early stopping patience for ensemble models.
        ensemble_num_layers: Number of hidden layers for ensemble subnets.
        ensemble_num_neurons: Number of neurons for ensemble subnets.
        baseline_epochs: Maximum epochs for baseline main effects model.
        baseline_patience: Early stopping patience for baseline model.
        interaction_epochs: Maximum epochs for interaction selection models.
        interaction_patience: Early stopping patience for interaction models.
        interaction_num_layers: Number of hidden layers for interaction subnets.
        interaction_num_neurons: Number of neurons for interaction subnets.
        batch_size: Batch size for training.
        learning_rate: Learning rate for training.
        init_bias: Whether to start each model's bias at the mean response.
        verbose: Whether to print progress messages.
    """
    num_ensemble: int = 10
    ensemble_epochs: int = 500
    ensemble_patience: int = 30
    ensemble_num_layers: int = 2
    ensemble_num_neurons: int = 20
    baseline_epochs: int = 5000
    baseline_patience: int = 50
    interaction_epochs: int = 500
    interaction_patience: int = 30
    interaction_num_layers: int = 2
    interaction_num_neurons: int = 40
    batch_size: int = 5000
    learning_rate: float = 0.001
    init_bias: bool = True
    verbose: bool = True


@dataclass
class SelectionResult:
    """Result of variable selection process.

    Attributes:
        selected_main_effects: List of selected main effect variable names.
        selected_interactions: List of selected (var1, var2) interaction tuples.
        main_importance: Dict mapping variable name to importance score.
        interaction_losses: Dict mapping (var1, var2) to validation loss.
        baseline_val_loss: Validation loss of main effects only model.
    """
    selected_main_effects: list[str] = field(default_factory=list)
    selected_interactions: list[tuple[str, str]] = field(default_factory=list)
    main_importance: dict[str, float] = field(default_factory=dict)
    interaction_losses: dict[tuple[str, str], float] = field(default_factory=dict)
    baseline_val_loss: float = 0.0


def train_ensemble_main_effects(
    feature_configs: dict,
    train_x: dict[str, Tensor],
    train_y: Tensor,
    val_x: dict[str, Tensor],
    val_y: Tensor,
    loss_fn: Callable = None,
    config: SparsityConfig = None,
    device: torch.device = None,
    train_offset: Tensor = None,
    val_offset: Tensor = None,
    base_seed: int = 0,
) -> list[dict[str, float]]:
    """
    Stage 1: Train an ensemble of NAMs with main effects only.

    Trains multiple NAMs with different random seeds to get stable
    estimates of variable importance.

    Args:
        feature_configs: Dict mapping feature name to config dict.
        train_x: Training inputs dict.
        train_y: Training targets.
        val_x: Validation inputs dict.
        val_y: Validation targets.
        loss_fn: Loss function (defaults to gamma_nll).
        config: SparsityConfig for training parameters.
        device: Torch device.
        base_seed: Starting seed for the ensemble; model ``i`` uses
            ``base_seed + i``. The default of ``0`` preserves the historical
            fixed seeds ``range(num_ensemble)`` so existing runs are unchanged.

    Returns:
        List of importance dicts, one per ensemble model.
    """
    config = config or SparsityConfig()
    loss_fn = loss_fn or gamma_nll
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_names = list(feature_configs.keys())
    ensemble_variances = []

    if config.verbose:
        print(f"Training ensemble of {config.num_ensemble} NAMs with "
              f"{len(feature_names)} main effects...")
        print("-" * 60)

    for i in range(config.num_ensemble):
        seed = base_seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model with main effects only
        model = ANAM(
            feature_configs=feature_configs,
            interactions=[],
            num_layers=config.ensemble_num_layers,
            num_neurons=config.ensemble_num_neurons,
            activation="leaky_relu",
            link="exp",
        )
        if config.init_bias:
            model.init_bias(train_y, train_offset)

        # Train
        train_config = TrainingConfig(
            epochs=config.ensemble_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            lambda_smooth=0.0,
            lambda_mc=0.0,
            patience=config.ensemble_patience,
            verbose=False,
        )

        train_anam(model, train_x, train_y, val_x, val_y,
                   train_offset=train_offset, val_offset=val_offset,
                   config=train_config, loss_fn=loss_fn, device=device)

        # Get variable importance
        importance = model.get_variable_importance(train_x)
        ensemble_variances.append(importance)

        if config.verbose:
            print(f"  Model {seed + 1}/{config.num_ensemble} trained")

    if config.verbose:
        print("-" * 60)
        print("Ensemble training complete!")

    return ensemble_variances


def compute_ensemble_importance(
    ensemble_variances: list[dict[str, float]],
) -> dict[str, float]:
    """
    Aggregate variable importance across ensemble models.

    Args:
        ensemble_variances: List of importance dicts from ensemble.

    Returns:
        Dict mapping variable name to average importance.
    """
    all_vars = list(ensemble_variances[0].keys())
    avg_importance = {}

    for var in all_vars:
        variances = [imp[var] for imp in ensemble_variances]
        avg_importance[var] = float(np.mean(variances))

    return avg_importance


def select_main_effects(
    importance: dict[str, float],
    k: int = None,
    threshold: float = None,
    min_ratio: float = None,
) -> list[str]:
    """
    Select important main effects based on ensemble variance.

    Three selection methods (use only one):
    1. Top K: Select the top k variables by importance.
    2. Threshold: Select variables with importance above threshold.
    3. Min ratio: Select variables with importance >= min_ratio * max_importance.

    Args:
        importance: Dict mapping variable name to importance score.
        k: Number of top variables to select.
        threshold: Minimum importance threshold.
        min_ratio: Minimum ratio relative to max importance.

    Returns:
        List of selected variable names, sorted by importance (descending).
    """
    # Sort by importance
    sorted_vars = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    if k is not None:
        return [v for v, _ in sorted_vars[:k]]
    elif threshold is not None:
        return [v for v, imp in sorted_vars if imp >= threshold]
    elif min_ratio is not None:
        max_imp = max(importance.values())
        return [v for v, imp in sorted_vars if imp >= min_ratio * max_imp]
    else:
        # Default: return all sorted by importance
        return [v for v, _ in sorted_vars]


def train_main_effects_model(
    feature_configs: dict,
    train_x: dict[str, Tensor],
    train_y: Tensor,
    val_x: dict[str, Tensor],
    val_y: Tensor,
    loss_fn: Callable = None,
    config: SparsityConfig = None,
    device: torch.device = None,
    seed: int = 42,
    train_offset: Tensor = None,
    val_offset: Tensor = None,
) -> tuple[ANAM, float]:
    """
    Train a main effects only model (baseline for Stage 2).

    Args:
        feature_configs: Feature configurations for selected variables.
        train_x: Training inputs dict.
        train_y: Training targets.
        val_x: Validation inputs dict.
        val_y: Validation targets.
        loss_fn: Loss function.
        config: SparsityConfig.
        device: Torch device.
        seed: Random seed.

    Returns:
        Tuple of (trained model, validation loss).
    """
    config = config or SparsityConfig()
    loss_fn = loss_fn or gamma_nll
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)

    model = ANAM(
        feature_configs=feature_configs,
        interactions=[],
        num_layers=config.ensemble_num_layers,
        num_neurons=config.ensemble_num_neurons,
        activation="leaky_relu",
        link="exp",
    )
    if config.init_bias:
        model.init_bias(train_y, train_offset)

    train_config = TrainingConfig(
        epochs=config.baseline_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        lambda_smooth=0.0,
        lambda_mc=0.0,
        patience=config.baseline_patience,
        verbose=config.verbose,
    )

    train_anam(model, train_x, train_y, val_x, val_y,
               train_offset=train_offset, val_offset=val_offset,
               config=train_config, loss_fn=loss_fn, device=device)

    # Compute validation loss
    model.eval()
    with torch.no_grad():
        val_pred = model(val_x, offset=val_offset)
        val_loss = loss_fn(val_y, val_pred).item()

    return model, val_loss


def evaluate_candidate_interactions(
    baseline_model: ANAM,
    feature_configs: dict,
    candidate_interactions: list[tuple[str, str]],
    train_x: dict[str, Tensor],
    train_y: Tensor,
    val_x: dict[str, Tensor],
    val_y: Tensor,
    loss_fn: Callable = None,
    config: SparsityConfig = None,
    device: torch.device = None,
    baseline_val_loss: float = None,
    train_offset: Tensor = None,
    val_offset: Tensor = None,
) -> dict[tuple[str, str], float]:
    """
    Stage 2: Evaluate candidate interactions by adding them one at a time.

    For each candidate interaction (var1, var2):
    1. Clone the baseline main effects model
    2. Add an interaction subnet for (var1, var2)
    3. Freeze all parameters EXCEPT:
       - The new interaction subnet
       - The main effect subnets for var1 and var2 (per the paper)
    4. Train and record validation loss

    Args:
        baseline_model: Trained main effects only model.
        feature_configs: Feature configurations.
        candidate_interactions: List of (var1, var2) tuples to evaluate.
        train_x: Training inputs dict.
        train_y: Training targets.
        val_x: Validation inputs dict.
        val_y: Validation targets.
        loss_fn: Loss function.
        config: SparsityConfig.
        device: Torch device.
        baseline_val_loss: Validation loss of baseline model (for logging).

    Returns:
        Dict mapping (var1, var2) to validation loss.
    """
    config = config or SparsityConfig()
    loss_fn = loss_fn or gamma_nll
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.verbose:
        print(f"Evaluating {len(candidate_interactions)} candidate interactions...")
        print("-" * 60)

    interaction_losses = {}

    for var1, var2 in candidate_interactions:
        torch.manual_seed(42)

        # Create new model with this interaction
        model = ANAM(
            feature_configs=feature_configs,
            interactions=[(var1, var2)],
            num_layers_main=config.ensemble_num_layers,
            num_neurons_main=config.ensemble_num_neurons,
            num_layers_interaction=config.interaction_num_layers,
            num_neurons_interaction=config.interaction_num_neurons,
            activation="leaky_relu",
            link="exp",
        )

        # Copy main effect weights from baseline model
        with torch.no_grad():
            for name, subnet in baseline_model.main_subnets.items():
                if name in model.main_subnets:
                    model.main_subnets[name].load_state_dict(subnet.state_dict())
            # Copy bias
            model.bias.copy_(baseline_model.bias)

        # Freeze main effect parameters EXCEPT for var1 and var2
        # This matches the paper: "only the parameters of the subnetwork or
        # lattice model corresponding to the pairwise term and its associated
        # main effects are trained"
        for name, subnet in model.main_subnets.items():
            if name not in (var1, var2):
                for param in subnet.parameters():
                    param.requires_grad = False

        # Train (only interaction subnet and var1, var2 main effects will update)
        train_config = TrainingConfig(
            epochs=config.interaction_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            lambda_smooth=0.0,
            lambda_mc=0.0,
            patience=config.interaction_patience,
            verbose=False,
        )

        train_anam(model, train_x, train_y, val_x, val_y,
                   train_offset=train_offset, val_offset=val_offset,
                   config=train_config, loss_fn=loss_fn, device=device)

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x, offset=val_offset)
            val_loss = loss_fn(val_y, val_pred).item()

        interaction_losses[(var1, var2)] = val_loss

        if config.verbose:
            improvement = (baseline_val_loss - val_loss) if baseline_val_loss else 0
            print(f"  {var1} x {var2}: val_loss={val_loss:.4f} "
                  f"(improvement: {improvement:+.4f})")

    if config.verbose:
        print("-" * 60)

    return interaction_losses


def select_interactions(
    interaction_losses: dict[tuple[str, str], float],
    baseline_val_loss: float,
    k: int = None,
    improvement_threshold: float = 0.0,
) -> list[tuple[str, str]]:
    """
    Select interactions that improve validation loss.

    Args:
        interaction_losses: Dict mapping (var1, var2) to validation loss.
        baseline_val_loss: Validation loss of main effects only model.
        k: Maximum number of interactions to select.
        improvement_threshold: Minimum improvement required (default 0).

    Returns:
        List of selected (var1, var2) tuples, sorted by improvement.
    """
    # Calculate improvements
    improvements = {
        pair: baseline_val_loss - loss
        for pair, loss in interaction_losses.items()
    }

    # Filter by threshold
    selected = [
        (pair, imp) for pair, imp in improvements.items()
        if imp > improvement_threshold
    ]

    # Sort by improvement (descending)
    selected.sort(key=lambda x: x[1], reverse=True)

    # Limit to top k
    if k is not None:
        selected = selected[:k]

    return [pair for pair, _ in selected]


def run_variable_selection(
    feature_configs: dict,
    train_x: dict[str, Tensor],
    train_y: Tensor,
    val_x: dict[str, Tensor],
    val_y: Tensor,
    loss_fn: Callable = None,
    config: SparsityConfig = None,
    device: torch.device = None,
    k_main: int = None,
    k_interaction: int = None,
    main_threshold: float = None,
    interaction_threshold: float = 0.0,
    train_offset: Tensor = None,
    val_offset: Tensor = None,
) -> SelectionResult:
    """
    Run the complete variable selection process (Stages 1 and 2).

    This is a convenience function that runs:
    1. Stage 1: Ensemble training for main effect importance
    2. Stage 2: Interaction selection with frozen main effects

    Args:
        feature_configs: Dict mapping feature name to config dict.
        train_x: Training inputs dict.
        train_y: Training targets.
        val_x: Validation inputs dict.
        val_y: Validation targets.
        loss_fn: Loss function (defaults to gamma_nll).
        config: SparsityConfig for training parameters.
        device: Torch device.
        k_main: Number of main effects to select (Stage 1).
        k_interaction: Number of interactions to select (Stage 2).
        main_threshold: Importance threshold for main effects.
        interaction_threshold: Minimum improvement for interactions.

    Returns:
        SelectionResult with selected features and diagnostics.
    """
    config = config or SparsityConfig()
    loss_fn = loss_fn or gamma_nll
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result = SelectionResult()

    # Stage 1: Main effect selection
    if config.verbose:
        print("=" * 60)
        print("STAGE 1: Main Effect Selection")
        print("=" * 60)

    ensemble_variances = train_ensemble_main_effects(
        feature_configs, train_x, train_y, val_x, val_y,
        loss_fn=loss_fn, config=config, device=device,
        train_offset=train_offset, val_offset=val_offset,
    )

    result.main_importance = compute_ensemble_importance(ensemble_variances)

    if k_main is not None:
        result.selected_main_effects = select_main_effects(
            result.main_importance, k=k_main
        )
    elif main_threshold is not None:
        result.selected_main_effects = select_main_effects(
            result.main_importance, threshold=main_threshold
        )
    else:
        # Default: select all
        result.selected_main_effects = select_main_effects(result.main_importance)

    if config.verbose:
        print(f"\nSelected {len(result.selected_main_effects)} main effects: "
              f"{result.selected_main_effects}")

    # Build feature configs for selected vars only
    selected_configs = {
        name: feature_configs[name]
        for name in result.selected_main_effects
    }

    # Filter train/val data to selected features
    selected_train_x = {k: v for k, v in train_x.items()
                        if k in result.selected_main_effects}
    selected_val_x = {k: v for k, v in val_x.items()
                      if k in result.selected_main_effects}

    # Stage 2: Interaction selection
    if config.verbose:
        print("\n" + "=" * 60)
        print("STAGE 2: Interaction Effect Selection")
        print("=" * 60)
        print("\nTraining baseline main effects model...")

    baseline_model, baseline_val_loss = train_main_effects_model(
        selected_configs, selected_train_x, train_y,
        selected_val_x, val_y,
        loss_fn=loss_fn, config=config, device=device,
        train_offset=train_offset, val_offset=val_offset,
    )
    result.baseline_val_loss = baseline_val_loss

    if config.verbose:
        print(f"\nBaseline validation loss: {baseline_val_loss:.4f}")
        print("\nEvaluating candidate interactions...")

    # Generate candidates (strong heredity)
    candidate_interactions = [
        (a, b) for a, b in combinations(result.selected_main_effects, 2)
        if interaction_supported(selected_configs[a], selected_configs[b])
    ]

    result.interaction_losses = evaluate_candidate_interactions(
        baseline_model, selected_configs, candidate_interactions,
        selected_train_x, train_y, selected_val_x, val_y,
        loss_fn=loss_fn, config=config, device=device,
        baseline_val_loss=baseline_val_loss,
        train_offset=train_offset, val_offset=val_offset,
    )

    result.selected_interactions = select_interactions(
        result.interaction_losses, baseline_val_loss,
        k=k_interaction, improvement_threshold=interaction_threshold
    )

    if config.verbose:
        print(f"\nSelected {len(result.selected_interactions)} interactions: "
              f"{result.selected_interactions}")
        print("\n" + "=" * 60)
        print("Variable selection complete!")
        print("=" * 60)

    return result


def plot_main_effect_importance(
    importance: dict[str, float],
    ax=None,
    title: str = "Main Effect Importance (Ensemble Variance)",
    top_k: int = None,
    cutoff: int = None,
):
    """
    Plot main effect importance as a bar chart.

    Args:
        importance: Dict mapping variable name to importance.
        ax: Matplotlib axis (creates new figure if None).
        title: Plot title.
        top_k: Only show top k variables.
        cutoff: If given, draw a dashed vertical marker after this many leading
            (highest-importance) bars to show the selection boundary.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    if top_k:
        sorted_items = sorted_items[:top_k]

    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    ax.bar(names, values)
    ax.set_xlabel("Variable")
    ax.set_ylabel("Subnet Variance")
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

    if cutoff is not None and 0 < cutoff <= len(names):
        ax.axvline(cutoff - 0.5, color="green", linestyle="--",
                   label=f"Selected ({cutoff})")
        ax.legend()

    return ax


def plot_interaction_selection(
    interaction_losses: dict[tuple[str, str], float],
    baseline_val_loss: float,
    ax=None,
    title: str = "Interaction Selection",
    top_k: int = 15,
    cutoff: int = None,
):
    """
    Plot interaction validation losses.

    Args:
        interaction_losses: Dict mapping (var1, var2) to validation loss.
        baseline_val_loss: Validation loss of main effects only model.
        ax: Matplotlib axis (creates new figure if None).
        title: Plot title.
        top_k: Number of interactions to show.
        cutoff: If given, draw a dashed vertical marker after this many leading
            (lowest-loss) bars to show how many interactions were selected.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4))

    # Sort by loss (ascending = best first) and plot each pair's *improvement*
    # over the mains-only baseline. Absolute losses differ by ~1e-4 on a ~0.4
    # scale, so bars-from-zero look identical; the improvement is the quantity
    # selection actually ranks on, and putting the baseline at 0 makes the
    # (small) differences and any pairs that hurt (negative bars) legible.
    sorted_items = sorted(interaction_losses.items(), key=lambda x: x[1])
    if top_k:
        sorted_items = sorted_items[:top_k]

    names = [f"{v1} x {v2}" for (v1, v2), _ in sorted_items]
    improvements = [baseline_val_loss - loss for _, loss in sorted_items]

    ax.bar(names, improvements)
    ax.axhline(0.0, color="red", linestyle="--", label="Main effects only")
    ax.set_xlabel("Interaction")
    ax.set_ylabel("Validation-loss improvement over baseline")
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

    if cutoff is not None and 0 < cutoff <= len(names):
        ax.axvline(cutoff - 0.5, color="green", linestyle="--",
                   label=f"Selected ({cutoff})")

    ax.legend()

    return ax
