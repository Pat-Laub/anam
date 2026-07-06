"""
Subnetwork modules for ANAM.

Each feature or interaction gets its own dedicated subnetwork (MLP)
to learn its shape function independently.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RunningMeanCenter(nn.Module):
    """
    Centers outputs by subtracting the running mean (without variance scaling).

    Unlike BatchNorm, this only centers (mean=0) without normalizing variance,
    preserving the magnitude of shape functions for interpretability.

    Args:
        num_features: Number of features (typically 1 for shape function output)
        momentum: Momentum for running mean update (default 0.1)
    """

    def __init__(self, num_features: int = 1, momentum: float = 0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        """
        Center the input by subtracting the mean.

        During training: updates running mean and centers using batch mean.
        During eval: centers using accumulated running mean.
        """
        if self.training:
            batch_mean = x.mean(dim=0)
            # Update running mean with exponential moving average. Detach so the
            # buffer never becomes part of the autograd graph: without this, the
            # update chains running_mean_n -> running_mean_{n-1} -> ... across
            # batches, retaining an ever-growing graph (a memory leak) and
            # leaving a buffer that erroneously carries grad. Centering still
            # backpropagates through the returned ``x - batch_mean``.
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * batch_mean
                )
                self.num_batches_tracked += 1
            return x - batch_mean
        else:
            return x - self.running_mean


class Subnet(nn.Module):
    """
    A feedforward neural network subnetwork for learning shape functions.

    Architecture:
        Input -> [Dense -> Activation -> Dropout] x num_layers -> Dense(1) -> Center -> Scale

    The centering layer subtracts the running mean to ensure
    identifiability of the shape function (mean=0). The output scale
    allows independent control of magnitude after centering.

    Args:
        input_dim: Number of input features (1 for main effects, 2 for interactions)
        num_layers: Number of hidden layers
        num_neurons: Number of neurons in first hidden layer (decreases linearly)
        activation: Activation function name ('leaky_relu', 'relu', 'tanh', 'elu')
        dropout: Dropout rate (0 to disable)
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_layers: int = 4,
        num_neurons: int = 100,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        # Build activation function
        self.activation_fn = self._get_activation(activation)

        # Build layers
        layers = []
        prev_size = input_dim

        for i in range(num_layers):
            # Linearly decreasing neuron count (like TF implementation)
            curr_size = num_neurons - i * (num_neurons // num_layers)
            curr_size = max(curr_size, 1)  # Ensure at least 1 neuron

            layers.append(nn.Linear(prev_size, curr_size))
            layers.append(self.activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = curr_size

        # Output layer (single value)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # Centering layer for identifiability (subtracts mean, no variance scaling)
        self.output_center = RunningMeanCenter(1)
        
        # Output scale (applied after centering to control magnitude independently)
        self.output_scale = nn.Parameter(torch.ones(1))

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the subnetwork.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, 1]
        """
        out = self.network(x)

        # Center output for identifiability
        out = self.output_center(out)
        
        # Scale output (independent magnitude control)
        out = out * self.output_scale

        return out


class EmbeddingSubnet(nn.Module):
    """
    Subnetwork for categorical features using embeddings.

    Embeds categorical features into a continuous space, then applies
    centering for identifiability and scaling for magnitude control.

    Args:
        num_categories: Number of unique categories
        embedding_dim: Dimension of embedding (default 1 for interpretability)
    """

    def __init__(self, num_categories: int, embedding_dim: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.output_center = RunningMeanCenter(embedding_dim)
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Integer tensor of category indices [batch_size] or [batch_size, 1]

        Returns:
            Embedded output [batch_size, embedding_dim]
        """
        if x.dim() == 2:
            x = x.squeeze(-1)
        x = x.long()
        out = self.embedding(x)

        # Center output for identifiability
        out = self.output_center(out)
        
        # Scale output (independent magnitude control)
        out = out * self.output_scale

        return out
