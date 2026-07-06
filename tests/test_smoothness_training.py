"""
Verify the smoothness + marginal-clarity penalties train without backward-graph
errors and that loss actually decreases over the short training run.
"""

import torch

from anam import (
    TrainingConfig,
    build_feature_configs,
    gamma_nll,
    generate_synthetic_data,
    prepare_data,
    train_anam,
)
from anam.model import ANAMModule


def test_smoothness_and_marginal_clarity_train_and_decrease():
    df = generate_synthetic_data(200, noise_level=1.0, random_state=42)

    features = ["X1", "X2", "X3"]
    configs = build_feature_configs(df, features)
    model = ANAMModule(
        configs, num_layers=2, num_neurons=20, interactions=[("X1", "X2")]
    )

    x, y = prepare_data(df, features, "Y")

    config = TrainingConfig(
        epochs=10,
        batch_size=50,
        verbose=False,
        lambda_smooth=1e-3,
        lambda_mc=1e-4,
        smooth_features=["X1", "X3"],
    )

    history = train_anam(model, x, y, config=config, loss_fn=gamma_nll)

    losses = history["train_loss"]
    assert losses[-1] < losses[0]
    # Sanity-check the history shape so silent flattening is caught.
    assert len(losses) == 10