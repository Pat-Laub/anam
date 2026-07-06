"""
Basic tests for ANAM package.
"""

import pytest
import torch
import numpy as np

from anam import (
    gamma_nll,
    poisson_nll,
    smoothness_penalty,
    generate_synthetic_data,
    build_feature_configs,
    train_anam,
    TrainingConfig,
    prepare_data,
    SparsityConfig,
    compute_ensemble_importance,
    select_main_effects,
    select_interactions,
)
from anam.model import ANAMModule as ANAM
from anam.lattice import project_monotonic_1d, dykstra_projection, PWLCalibration, MonotonicSubnet
from anam.subnet import Subnet, RunningMeanCenter


class TestLossFunctions:
    """Tests for loss functions."""

    def test_gamma_nll_positive(self):
        """Gamma NLL should be positive for reasonable inputs."""
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        loss = gamma_nll(y_true, y_pred)
        assert loss.item() >= 0

    def test_gamma_nll_perfect_prediction(self):
        """Gamma NLL should be 1 when prediction equals truth.

        gamma_nll = mean(y/y_pred - log(y/y_pred))
        When y == y_pred: mean(1 - log(1)) = mean(1 - 0) = 1
        """
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        loss = gamma_nll(y_true, y_pred)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_poisson_nll_positive(self):
        """Poisson NLL should be positive."""
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        loss = poisson_nll(y_true, y_pred)
        assert loss.item() >= 0


class TestMonotonicity:
    """Tests for monotonicity projection."""

    def test_project_1d_increasing(self):
        """1D projection should produce increasing sequence."""
        params = torch.tensor([3.0, 1.0, 2.0, 0.0, 4.0])
        result = project_monotonic_1d(params, "increasing")

        # Check monotonicity
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] + 1e-6

    def test_project_1d_decreasing(self):
        """1D projection should produce decreasing sequence."""
        params = torch.tensor([0.0, 2.0, 1.0, 3.0, -1.0])
        result = project_monotonic_1d(params, "decreasing")

        # Check monotonicity
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1] - 1e-6

    def test_project_2d_increasing(self):
        """2D Dykstra projection should produce monotonic lattice."""
        torch.manual_seed(42)
        params = torch.randn(4 * 4)  # 4x4 lattice
        result = dykstra_projection(params, 4, 4, "increasing", "increasing")

        # Reshape and check monotonicity
        # Use 1e-5 tolerance to account for numerical precision in iterative algorithm
        lattice = result.view(4, 4)
        for i in range(3):
            for j in range(4):
                assert lattice[i, j] <= lattice[i + 1, j] + 1e-5
        for i in range(4):
            for j in range(3):
                assert lattice[i, j] <= lattice[i, j + 1] + 1e-5


class TestRunningMeanCenter:
    """Regression tests for the centering layer used by every subnet."""

    def test_running_mean_buffer_stays_detached(self):
        """The running-mean buffer must never enter the autograd graph.

        Regression test for the EMA update, which used to keep ``batch_mean``
        (which requires grad) attached to the buffer. That chained
        running_mean_n -> running_mean_{n-1} -> ... across batches, retaining an
        ever-growing graph (a memory leak) and leaving a buffer that
        erroneously carried grad. The buffer must stay a detached leaf.
        """
        net = Subnet(input_dim=1, num_layers=2, num_neurons=8)
        net.train()
        # Several batches: the pre-fix leak would deepen the retained graph on
        # each pass, so backward is exercised repeatedly.
        for _ in range(5):
            x = torch.randn(16, 1, requires_grad=True)
            loss = (net(x) ** 2).mean()
            loss.backward()

        rm = net.output_center.running_mean
        assert not rm.requires_grad
        assert rm.grad_fn is None
        assert rm.is_leaf

    def test_centering_still_backpropagates(self):
        """Detaching the buffer update must not stop gradients reaching params.

        Centering backpropagates through the returned ``x - batch_mean``, so the
        network's weights should still receive gradients.
        """
        net = Subnet(input_dim=1, num_layers=2, num_neurons=8)
        net.train()
        x = torch.randn(16, 1, requires_grad=True)
        loss = (net(x) ** 2).mean()
        loss.backward()
        assert all(p.grad is not None for p in net.network.parameters())

    def test_running_mean_tracks_data_and_centers_in_eval(self):
        """The EMA still converges to the data mean and centers at eval time."""
        center = RunningMeanCenter(1, momentum=0.5)
        center.train()
        data = torch.full((256, 1), 3.0)
        for _ in range(20):
            center(data)

        # EMA has converged to the (constant) batch mean.
        assert center.running_mean.item() == pytest.approx(3.0, abs=1e-3)

        # Eval subtracts the running mean, so a constant input centers to ~0.
        center.eval()
        with torch.no_grad():
            out = center(data)
        assert out.abs().max().item() == pytest.approx(0.0, abs=1e-3)


class TestMonotonicityConstraint:
    """The imposed monotonicity must hold even when the data contradicts it.

    A monotonic subnet has a free ``output_scale`` multiplier; a negative scale
    could in principle invert the sign and defeat the imposed direction. These
    tests pin down that the constraint is never violated in practice, while the
    subnet remains capable of fitting a compatible (increasing) target.
    """

    def _fit_scale_subnet(self, target_fn, steps=500, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        sub = MonotonicSubnet(
            input_bounds={'x': (-2.0, 2.0)},
            monotonicity={'x': 'increasing'},
        )
        opt = torch.optim.Adam(sub.parameters(), lr=0.05)
        xs = torch.linspace(-2, 2, 256)
        target = target_fn(xs)
        for _ in range(steps):
            opt.zero_grad()
            out = sub({'x': xs}).squeeze(-1)
            ((out - target) ** 2).mean().backward()
            opt.step()
            sub.project_monotonicity()
        sub.eval()
        with torch.no_grad():
            grid = torch.linspace(-2, 2, 60)
            vals = sub({'x': grid}).squeeze(-1).numpy()
        return sub, vals

    def test_increasing_constraint_holds_against_decreasing_target(self):
        """Fitting a strongly decreasing target must not produce a decreasing
        shape function: the constraint wins (the lattice flattens rather than
        the output scale flipping sign)."""
        _, vals = self._fit_scale_subnet(lambda xs: -3.0 * xs)
        assert np.all(np.diff(vals) >= -1e-4)

    def test_increasing_constraint_fits_increasing_target(self):
        """Sanity check that the constraint does not hold merely because the
        subnet is inert: given a compatible increasing target it learns a
        genuinely increasing (non-flat) function."""
        _, vals = self._fit_scale_subnet(lambda xs: 3.0 * xs)
        assert np.all(np.diff(vals) >= -1e-4)      # still monotone increasing
        assert vals[-1] - vals[0] > 1.0            # and genuinely non-flat


class TestPWLCalibration:
    """Tests for PWL calibration layer."""

    def test_output_range_default(self):
        """Default output range should be [0, num_keypoints-1]."""
        calibrator = PWLCalibration(-1.0, 1.0, num_keypoints=10)

        inputs = torch.linspace(-1.0, 1.0, 100)
        outputs = calibrator(inputs)

        assert outputs.min().item() >= 0.0
        assert outputs.max().item() <= 9.0  # num_keypoints - 1

    def test_output_range_custom(self):
        """Custom output_max should scale output range correctly."""
        num_keypoints = 20
        num_vertices = 10

        calibrator = PWLCalibration(
            -1.0, 1.0,
            num_keypoints=num_keypoints,
            output_max=float(num_vertices - 1)  # Should be [0, 9]
        )

        inputs = torch.linspace(-1.0, 1.0, 100)
        outputs = calibrator(inputs)

        # Output should be in [0, num_vertices-1], not [0, num_keypoints-1]
        assert outputs.min().item() >= 0.0
        assert outputs.max().item() <= float(num_vertices - 1) + 1e-5
        # Most importantly, output should NOT go up to num_keypoints-1
        assert outputs.max().item() < num_keypoints - 1

    def test_monotonic_subnet_output_range(self):
        """MonotonicSubnet should align calibrator output with lattice input."""
        num_keypoints = 20
        num_vertices = 10

        subnet = MonotonicSubnet(
            input_bounds={'X': (-1.0, 1.0)},
            monotonicity={'X': 'increasing'},
            num_keypoints=num_keypoints,
            num_vertices=num_vertices,
        )

        # Get the calibrator's output range
        calibrator = subnet.calibrators['X']
        inputs = torch.linspace(-1.0, 1.0, 100)
        calibrated = calibrator(inputs)

        # Calibrator should output in [0, num_vertices-1], not [0, num_keypoints-1]
        assert calibrated.max().item() <= float(num_vertices - 1) + 1e-5
        assert calibrated.max().item() < num_keypoints - 1


class TestSmoothnesspenalty:
    """Tests for smoothness penalty."""

    def test_smooth_function_low_penalty(self):
        """Smooth function should have low penalty."""
        grid = torch.linspace(0, 1, 100)
        smooth_values = grid ** 2  # Smooth parabola
        h = 1.0 / 99
        penalty = smoothness_penalty(smooth_values, h, lambda_smooth=1.0)
        assert penalty.item() < 10  # Should be small for smooth function

    def test_rough_function_high_penalty(self):
        """Rough function should have higher penalty."""
        grid = torch.linspace(0, 1, 100)
        rough_values = torch.sin(50 * grid)  # High frequency oscillation
        h = 1.0 / 99
        penalty = smoothness_penalty(rough_values, h, lambda_smooth=1.0)
        assert penalty.item() > 100  # Should be large for rough function


class TestModel:
    """Tests for ANAM model."""

    def test_model_creation(self):
        """Model should be created with given configs."""
        configs = {
            'X1': {'type': 'continuous', 'bounds': (-1, 1)},
            'X2': {'type': 'continuous', 'bounds': (-1, 1), 'monotonicity': 'increasing'},
        }
        model = ANAM(configs, interactions=[('X1', 'X2')])

        assert 'X1' in model.main_subnets
        assert 'X2' in model.main_subnets
        assert 'X1_X2' in model.interaction_subnets

    def test_forward_pass(self):
        """Forward pass should produce correct output shape."""
        configs = {
            'X1': {'type': 'continuous', 'bounds': (-1, 1)},
            'X2': {'type': 'continuous', 'bounds': (-1, 1)},
        }
        model = ANAM(configs)

        x = {
            'X1': torch.randn(32),
            'X2': torch.randn(32),
        }
        output = model(x)

        assert output.shape == (32, 1)
        assert (output > 0).all()  # Exponential link ensures positive

    def test_shape_function_extraction(self):
        """Shape function should be extractable."""
        configs = {
            'X1': {'type': 'continuous', 'bounds': (-1, 1)},
        }
        model = ANAM(configs)

        grid = torch.linspace(-1, 1, 50)
        shape = model.get_shape_function('X1', grid)

        assert shape.shape == (50,)

    def test_components_consistent_across_methods(self):
        """forward(return_components) must agree with the per-term accessors.

        Guards the shared ``_eval_main_subnet``/``_eval_interaction_subnet``
        dispatch: forward, get_main_outputs and get_interaction_outputs are three
        entry points into the same subnet evaluation and must never diverge.
        Uses a mix of subnet types (plain MLP, categorical embedding, monotonic
        lattice) so every dispatch branch is exercised.
        """
        configs = {
            'X1': {'type': 'continuous', 'bounds': (-1, 1)},
            'X2': {'type': 'continuous', 'bounds': (-1, 1), 'monotonicity': 'increasing'},
            'C': {'type': 'categorical', 'num_categories': 4},
        }
        model = ANAM(configs, interactions=[('X1', 'X2')])
        model.eval()  # deterministic centering (running mean, no dropout)

        x = {
            'X1': torch.randn(16),
            'X2': torch.randn(16),
            'C': torch.randint(0, 4, (16,)),
        }

        with torch.no_grad():
            pred, mains, inters = model(x, return_components=True)
            main_only = model.get_main_outputs(x)
            inter_only = model.get_interaction_outputs(x)

        for name in configs:
            assert torch.equal(mains[name], main_only[name])
        assert torch.equal(inters[('X1', 'X2')], inter_only[('X1', 'X2')])

        # Components (plus bias) must reconstruct the linear predictor exactly.
        total = model.bias.clone()
        for out in list(mains.values()) + list(inters.values()):
            total = total + out
        assert torch.allclose(pred, torch.exp(total), atol=1e-6)

    def test_interaction_shape_function_returns_canonical_order(self):
        """get_shape_function evaluates an interaction in its stored key order.

        When queried with the features reversed, the engine swaps to the stored
        key internally and returns the same canonical surface (it does not
        re-transpose to the requested order — that is the wrapper's job, see
        tests/test_api.py::TestEffect). Locking this pins the contract the
        wrapper's transpose compensation relies on.
        """
        configs = {
            'A': {'type': 'continuous', 'bounds': (-1, 1)},
            'B': {'type': 'continuous', 'bounds': (-1, 1)},
        }
        model = ANAM(configs, interactions=[('A', 'B')])
        model.eval()

        grid_a = torch.linspace(-1, 1, 7)
        grid_b = torch.linspace(-1, 1, 5)
        with torch.no_grad():
            vals_ab = model.get_shape_function('A', grid_a, 'B', grid_b)
            vals_ba = model.get_shape_function('B', grid_b, 'A', grid_a)

        # Both come back in canonical [A, B] orientation and are identical.
        assert vals_ab.shape == (7, 5)
        assert vals_ba.shape == (7, 5)
        assert torch.allclose(vals_ab, vals_ba, atol=1e-6)


class TestTraining:
    """Tests for training functionality."""

    def test_prepare_data(self):
        """Data preparation should work correctly."""
        import pandas as pd
        df = pd.DataFrame({
            'X1': np.random.randn(100),
            'X2': np.random.randn(100),
            'Y': np.abs(np.random.randn(100)) + 0.1,
        })

        x, y = prepare_data(df, ['X1', 'X2'], 'Y')

        assert 'X1' in x
        assert 'X2' in x
        assert x['X1'].shape == (100,)
        assert y.shape == (100, 1)

    def test_prepare_data_with_offset(self):
        """Data preparation with offset should return three elements."""
        import pandas as pd
        df = pd.DataFrame({
            'X1': np.random.randn(100),
            'X2': np.random.randn(100),
            'exposure': np.random.uniform(0.5, 1.0, 100),
            'Y': np.abs(np.random.randn(100)) + 0.1,
        })

        x, y, offset = prepare_data(df, ['X1', 'X2'], 'Y', offset='exposure')

        assert 'X1' in x
        assert 'X2' in x
        assert x['X1'].shape == (100,)
        assert y.shape == (100, 1)
        assert offset.shape == (100,)
        # Check that offset is log-transformed by default
        assert torch.allclose(offset, torch.log(torch.tensor(df['exposure'].values, dtype=torch.float32)))

    def test_forward_with_offset(self):
        """Forward pass with offset should produce different results."""
        configs = {
            'X1': {'type': 'continuous', 'bounds': (-1, 1)},
            'X2': {'type': 'continuous', 'bounds': (-1, 1)},
        }
        model = ANAM(configs)

        x = {
            'X1': torch.randn(32),
            'X2': torch.randn(32),
        }

        # Without offset
        output_no_offset = model(x)

        # With offset (should multiply predictions by exp(offset))
        offset = torch.ones(32) * 0.5  # log(exposure) = 0.5
        output_with_offset = model(x, offset=offset)

        # With positive offset and exp link, outputs with offset should be larger
        assert (output_with_offset > output_no_offset).all()
        # Check the relationship: output_with_offset = output_no_offset * exp(offset)
        expected = output_no_offset * torch.exp(offset).unsqueeze(-1)
        assert torch.allclose(output_with_offset, expected, rtol=1e-5)

    def test_training_reduces_loss(self):
        """Training should reduce loss over epochs."""
        # Generate small dataset
        df = generate_synthetic_data(500, noise_level=1.0, random_state=42)

        features = ['X1', 'X2']
        configs = build_feature_configs(df, features)
        model = ANAM(configs, num_layers=2, num_neurons=20)

        x, y = prepare_data(df, features, 'Y')

        # Initial loss
        with torch.no_grad():
            initial_pred = model(x)
            initial_loss = gamma_nll(y, initial_pred)

        # Train
        config = TrainingConfig(epochs=50, batch_size=100, verbose=False)
        history = train_anam(model, x, y, config=config)

        # Check loss decreased
        assert history['train_loss'][-1] < initial_loss.item()


class TestSparsity:
    """Tests for sparsity/variable selection functionality."""

    def test_compute_ensemble_importance(self):
        """Ensemble importance should average across models."""
        ensemble_variances = [
            {'X1': 0.5, 'X2': 0.3, 'X3': 0.1},
            {'X1': 0.6, 'X2': 0.2, 'X3': 0.2},
            {'X1': 0.4, 'X2': 0.4, 'X3': 0.1},
        ]

        importance = compute_ensemble_importance(ensemble_variances)

        assert importance['X1'] == pytest.approx(0.5, abs=1e-6)
        assert importance['X2'] == pytest.approx(0.3, abs=1e-6)
        assert importance['X3'] == pytest.approx(0.133333, abs=1e-3)

    def test_select_main_effects_top_k(self):
        """Should select top k variables by importance."""
        importance = {'X1': 0.5, 'X2': 0.3, 'X3': 0.1, 'X4': 0.4}

        selected = select_main_effects(importance, k=2)

        assert len(selected) == 2
        assert selected[0] == 'X1'  # Highest
        assert selected[1] == 'X4'  # Second highest

    def test_select_main_effects_threshold(self):
        """Should select variables above threshold."""
        importance = {'X1': 0.5, 'X2': 0.3, 'X3': 0.1, 'X4': 0.4}

        selected = select_main_effects(importance, threshold=0.35)

        assert len(selected) == 2
        assert 'X1' in selected
        assert 'X4' in selected

    def test_select_interactions_improvement(self):
        """Should select interactions that improve validation loss."""
        interaction_losses = {
            ('X1', 'X2'): 0.40,  # Improvement of 0.1
            ('X1', 'X3'): 0.52,  # No improvement (worse)
            ('X2', 'X3'): 0.45,  # Improvement of 0.05
        }
        baseline_loss = 0.50

        selected = select_interactions(interaction_losses, baseline_loss)

        assert len(selected) == 2
        assert ('X1', 'X2') == selected[0]  # Best improvement
        assert ('X2', 'X3') == selected[1]

    def test_select_interactions_with_k(self):
        """Should limit selection to top k."""
        interaction_losses = {
            ('X1', 'X2'): 0.40,
            ('X1', 'X3'): 0.42,
            ('X2', 'X3'): 0.45,
        }
        baseline_loss = 0.50

        selected = select_interactions(interaction_losses, baseline_loss, k=1)

        assert len(selected) == 1
        assert selected[0] == ('X1', 'X2')

    def test_sparsity_config_defaults(self):
        """SparsityConfig should have sensible defaults."""
        config = SparsityConfig()

        assert config.num_ensemble == 10
        assert config.ensemble_epochs == 500
        assert config.batch_size == 5000
        assert config.verbose is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
