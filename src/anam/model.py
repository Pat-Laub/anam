"""
ANAM - Actuarial Neural Additive Model

Main model class that combines subnetworks for main effects and
interactions into an interpretable additive model.
"""

import torch
import torch.nn as nn
from torch import Tensor

from anam.subnet import Subnet, EmbeddingSubnet
from anam.lattice import MonotonicSubnet


class ANAMModule(nn.Module):
    """
    Actuarial Neural Additive Model (the raw ``nn.Module`` engine).

    An interpretable deep learning model that learns additive shape functions
    for each feature and pairwise interaction. Supports:
    - Continuous and categorical features
    - Monotonicity constraints (via lattice regression)
    - Exponential output link function (for Gamma/Poisson regression)

    The model structure is:
        g(mu) = bias + sum(s_i(X_i)) + sum(h_ij(X_i, X_j))

    where s_i are main effect shape functions and h_ij are interaction functions.

    Args:
        feature_configs: Dict mapping feature name to config dict with keys:
            - 'type': 'continuous' or 'categorical'
            - 'bounds': (min, max) for continuous features
            - 'num_categories': int for categorical features
            - 'monotonicity': 'increasing', 'decreasing', or None
        interactions: List of (feature1, feature2) tuples for interaction terms
        num_layers: Number of hidden layers for MLP subnets
        num_neurons: Number of neurons in first layer of MLP subnets
        num_keypoints: Number of keypoints for PWL calibration
        num_vertices: Number of lattice vertices for monotonic features
        activation: Activation function for MLP subnets
        dropout: Dropout rate for MLP subnets
        link: Output link function ('exp' for exponential, 'identity' for linear)
    """

    def __init__(
        self,
        feature_configs: dict,
        interactions: list[tuple[str, str]] = None,
        num_layers: int = 4,
        num_neurons: int = 100,
        num_layers_main: int = None,
        num_neurons_main: int = None,
        num_layers_interaction: int = None,
        num_neurons_interaction: int = None,
        num_keypoints: int = 20,
        num_vertices: int = 10,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        link: str = "exp",
    ):
        super().__init__()

        self.feature_configs = feature_configs
        self.feature_names = list(feature_configs.keys())
        self.interactions = interactions or []
        self.link = link
        self.num_keypoints = num_keypoints
        self.num_vertices = num_vertices

        # Use separate params if provided, otherwise fall back to shared params
        self.num_layers_main = num_layers_main if num_layers_main is not None else num_layers
        self.num_neurons_main = num_neurons_main if num_neurons_main is not None else num_neurons
        self.num_layers_interaction = num_layers_interaction if num_layers_interaction is not None else num_layers
        self.num_neurons_interaction = num_neurons_interaction if num_neurons_interaction is not None else num_neurons

        # Identify monotonic features
        self.monotonic_features = {
            name: cfg.get('monotonicity')
            for name, cfg in feature_configs.items()
            if cfg.get('monotonicity') is not None
        }

        # Build main effect subnets
        self.main_subnets = nn.ModuleDict()
        for name, config in feature_configs.items():
            subnet = self._build_main_subnet(
                name, config,
                self.num_layers_main, self.num_neurons_main,
                activation, dropout
            )
            self.main_subnets[name] = subnet

        # Build interaction subnets
        self.interaction_subnets = nn.ModuleDict()
        for var1, var2 in self.interactions:
            key = f"{var1}_{var2}"
            subnet = self._build_interaction_subnet(
                var1, var2,
                self.num_layers_interaction, self.num_neurons_interaction,
                activation, dropout
            )
            self.interaction_subnets[key] = subnet

        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))

    def init_bias(self, y: Tensor, offset: Tensor = None):
        """Set the bias to the intercept-only fit of ``y`` (and the offset)."""
        if self.link == "exp":
            exposure = torch.exp(offset).sum() if offset is not None else y.numel()
            bias = torch.log(y.sum() / exposure)
        else:
            bias = y.mean() - (offset.mean() if offset is not None else 0.0)
        if torch.isfinite(bias):
            with torch.no_grad():
                self.bias.fill_(bias.item())

    def _build_main_subnet(
        self, name: str, config: dict,
        num_layers: int, num_neurons: int,
        activation: str, dropout: float,
    ) -> nn.Module:
        """Build a subnetwork for a main effect."""
        feat_type = config.get('type', 'continuous')
        monotonicity = config.get('monotonicity')

        if feat_type == 'categorical':
            num_categories = config['num_categories']
            return EmbeddingSubnet(num_categories)

        elif monotonicity is not None:
            # Use lattice for monotonic continuous features
            bounds = config['bounds']
            return MonotonicSubnet(
                input_bounds={name: bounds},
                monotonicity={name: monotonicity},
                num_keypoints=self.num_keypoints,
                num_vertices=self.num_vertices,
            )
        else:
            # Use MLP for non-monotonic continuous features
            return Subnet(
                input_dim=1,
                num_layers=num_layers,
                num_neurons=num_neurons,
                activation=activation,
                dropout=dropout,
            )

    def _build_interaction_subnet(
        self, var1: str, var2: str,
        num_layers: int, num_neurons: int,
        activation: str, dropout: float,
    ) -> nn.Module:
        """Build a subnetwork for an interaction effect."""
        config1 = self.feature_configs[var1]
        config2 = self.feature_configs[var2]

        # Check if any variable needs monotonicity
        mono1 = config1.get('monotonicity')
        mono2 = config2.get('monotonicity')

        if mono1 is not None or mono2 is not None:
            # Use 2D lattice for monotonic interactions
            bounds = {}
            monotonicity = {}

            if config1.get('type', 'continuous') == 'continuous':
                bounds[var1] = config1['bounds']
                if mono1:
                    monotonicity[var1] = mono1

            if config2.get('type', 'continuous') == 'continuous':
                bounds[var2] = config2['bounds']
                if mono2:
                    monotonicity[var2] = mono2

            return MonotonicSubnet(
                input_bounds=bounds,
                monotonicity=monotonicity,
                num_keypoints=self.num_keypoints,
                num_vertices=self.num_vertices,
            )
        else:
            # Use MLP for non-monotonic interactions
            return Subnet(
                input_dim=2,
                num_layers=num_layers,
                num_neurons=num_neurons,
                activation=activation,
                dropout=dropout,
            )

    def _eval_main_subnet(self, name: str, feat: Tensor) -> Tensor:
        """Evaluate a single main-effect subnet, dispatching on its type.

        Centralises the ``MonotonicSubnet``/``EmbeddingSubnet``/``Subnet``
        dispatch (and the ``[batch] -> [batch, 1]`` unsqueeze for MLP subnets)
        so :meth:`forward` and :meth:`get_main_outputs` share one code path.
        """
        subnet = self.main_subnets[name]
        if isinstance(subnet, MonotonicSubnet):
            return subnet({name: feat})
        if isinstance(subnet, EmbeddingSubnet):
            return subnet(feat)
        if feat.dim() == 1:
            feat = feat.unsqueeze(-1)
        return subnet(feat)

    def _eval_interaction_subnet(
        self, var1: str, var2: str, x: dict[str, Tensor]
    ) -> Tensor:
        """Evaluate a single interaction subnet, dispatching on its type.

        Shared by :meth:`forward` and :meth:`get_interaction_outputs`.
        """
        subnet = self.interaction_subnets[f"{var1}_{var2}"]
        if isinstance(subnet, MonotonicSubnet):
            return subnet({var1: x[var1], var2: x[var2]})
        feat1 = x[var1]
        feat2 = x[var2]
        if feat1.dim() == 1:
            feat1 = feat1.unsqueeze(-1)
        if feat2.dim() == 1:
            feat2 = feat2.unsqueeze(-1)
        return subnet(torch.cat([feat1, feat2], dim=-1))

    def forward(
        self,
        x: dict[str, Tensor],
        offset: Tensor = None,
        return_components: bool = False,
    ):
        """
        Forward pass through the model.

        Args:
            x: Dict mapping feature name to tensor [batch_size] or [batch_size, 1]
            offset: Optional offset term [batch_size] or [batch_size, 1] to add to the
                linear predictor before applying the link function. For Poisson regression
                with exposure, pass log(exposure) as offset. This implements:
                    g(μ) = bias + Σ s_i(X_i) + Σ h_ij(X_i, X_j) + offset
            return_components: If True, also return per-feature and per-interaction
                               contributions as dictionaries.

        Returns:
            If return_components is False (default):
                Predictions [batch_size, 1]
            If True:
                (predictions, main_outputs, interaction_outputs)

                main_outputs: dict[str, Tensor] with shape [batch_size, 1]
                interaction_outputs: dict[(str, str), Tensor] with shape [batch_size, 1]
        """
        main_outputs: dict[str, Tensor] = {}
        interaction_outputs: dict[tuple[str, str], Tensor] = {}
        contributions = []

        # ----- Main effects -----
        for name in self.feature_names:
            out = self._eval_main_subnet(name, x[name])
            contributions.append(out)

            if return_components:
                main_outputs[name] = out

        # ----- Interaction effects -----
        for var1, var2 in self.interactions:
            out = self._eval_interaction_subnet(var1, var2, x)
            contributions.append(out)

            if return_components:
                interaction_outputs[(var1, var2)] = out

        # Sum all contributions
        combined = torch.stack(contributions, dim=-1).sum(dim=-1)
        combined = combined + self.bias

        # Add offset if provided (e.g., log(exposure) for Poisson regression)
        if offset is not None:
            if offset.dim() == 1:
                offset = offset.unsqueeze(-1)
            combined = combined + offset

        # Apply link function
        if self.link == "exp":
            pred = torch.exp(combined)
        else:
            pred = combined

        if return_components:
            return pred, main_outputs, interaction_outputs

        return pred


    def get_main_outputs(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Get individual main effect outputs (for visualization and penalties).

        Args:
            x: Input dict

        Returns:
            Dict mapping feature name to shape function output [batch_size, 1]
        """
        outputs = {}
        for name in self.feature_names:
            outputs[name] = self._eval_main_subnet(name, x[name])

        return outputs

    def get_interaction_outputs(self, x: dict[str, Tensor]) -> dict[tuple[str, str], Tensor]:
        """
        Get individual interaction effect outputs.

        Args:
            x: Input dict

        Returns:
            Dict mapping (var1, var2) to interaction output [batch_size, 1]
        """
        outputs = {}
        for var1, var2 in self.interactions:
            outputs[(var1, var2)] = self._eval_interaction_subnet(var1, var2, x)

        return outputs

    def get_shape_function(
        self,
        feature: str,
        grid: Tensor,
        other_feature: str = None,
        other_grid: Tensor = None,
    ) -> Tensor:
        """
        Evaluate shape function on a grid for visualization.

        For main effects, provide only feature and grid.
        For interactions, provide both feature pairs and grids.

        Note: Call with model.eval() for deterministic shapes (no dropout,
        centering uses running mean).

        Args:
            feature: Feature name
            grid: Grid values [N] for the feature
            other_feature: Second feature name (for interactions)
            other_grid: Grid values [M] for second feature

        Returns:
            Shape function values [N] for main effects or [N, M] for interactions
        """
        if other_feature is None:
            # Main effect
            subnet = self.main_subnets[feature]

            if isinstance(subnet, MonotonicSubnet):
                out = subnet({feature: grid})
            elif isinstance(subnet, EmbeddingSubnet):
                out = subnet(grid)
            else:
                out = subnet(grid.unsqueeze(-1))

            return out.squeeze(-1)

        else:
            # Interaction effect
            key = f"{feature}_{other_feature}"
            if key not in self.interaction_subnets:
                key = f"{other_feature}_{feature}"
                feature, other_feature = other_feature, feature
                grid, other_grid = other_grid, grid

            subnet = self.interaction_subnets[key]

            # Create meshgrid
            N, M = len(grid), len(other_grid)
            g1, g2 = torch.meshgrid(grid, other_grid, indexing='ij')
            g1_flat = g1.reshape(-1)
            g2_flat = g2.reshape(-1)

            if isinstance(subnet, MonotonicSubnet):
                out = subnet({feature: g1_flat, other_feature: g2_flat})
            else:
                combined = torch.stack([g1_flat, g2_flat], dim=-1)
                out = subnet(combined)

            return out.squeeze(-1).reshape(N, M)

    def project_monotonicity(self):
        """Project all monotonic components to satisfy constraints."""
        for subnet in self.main_subnets.values():
            if isinstance(subnet, MonotonicSubnet):
                subnet.project_monotonicity()

        for subnet in self.interaction_subnets.values():
            if isinstance(subnet, MonotonicSubnet):
                subnet.project_monotonicity()

    def get_variable_importance(self, x: dict[str, Tensor]) -> dict[str, float]:
        """
        Compute variable importance as variance of shape functions.

        Args:
            x: Input data dict

        Returns:
            Dict mapping feature/interaction name to importance score
        """
        self.eval()
        importance = {}

        with torch.no_grad():
            # Main effects
            main_outputs = self.get_main_outputs(x)
            for name, out in main_outputs.items():
                importance[name] = out.var().item()

            # Interactions
            int_outputs = self.get_interaction_outputs(x)
            for (var1, var2), out in int_outputs.items():
                importance[f"{var1}_{var2}"] = out.var().item()

        return importance


# Backward-compatibility alias: the raw engine used to be exported as ``ANAM``.
# Old code importing ``from anam.model import ANAM`` keeps working. The
# package-level ``anam.ANAM`` name is the user-facing wrapper (see api.py).
ANAM = ANAMModule
