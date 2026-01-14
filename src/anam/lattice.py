"""
Lattice regression modules for monotonicity constraints.

Implements monotonic lattice regression following Garcia & Gupta (2009)
and Gupta et al. (2016), with Dykstra's projection algorithm for
enforcing monotonicity constraints.
"""

import torch
import torch.nn as nn
from torch import Tensor

from anam.subnet import RunningMeanCenter


def dykstra_projection(
    params: Tensor,
    num_vertices_dim1: int,
    num_vertices_dim2: int,
    monotonicity_dim1: str = "none",
    monotonicity_dim2: str = "none",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tensor:
    """
    Project lattice parameters onto the feasible set defined by monotonicity
    constraints using Dykstra's projection algorithm.

    For a 2D lattice with shape [M1, M2], monotonicity constraints are:
        - dim1 increasing: lambda[a+1, b] >= lambda[a, b]
        - dim1 decreasing: lambda[a+1, b] <= lambda[a, b]
        - dim2 increasing: lambda[a, b+1] >= lambda[a, b]
        - dim2 decreasing: lambda[a, b+1] <= lambda[a, b]

    Args:
        params: Lattice parameters [M1 * M2] (flattened)
        num_vertices_dim1: Number of vertices along dimension 1
        num_vertices_dim2: Number of vertices along dimension 2
        monotonicity_dim1: Monotonicity for dim 1 ("increasing", "decreasing", "none")
        monotonicity_dim2: Monotonicity for dim 2 ("increasing", "decreasing", "none")
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Projected parameters [M1 * M2]
    """
    if monotonicity_dim1 == "none" and monotonicity_dim2 == "none":
        return params

    M1, M2 = num_vertices_dim1, num_vertices_dim2
    x = params.clone().view(M1, M2)

    # Build constraint index tensors for vectorized operations
    i1_list, j1_list, i2_list, j2_list = [], [], [], []

    if monotonicity_dim1 == "increasing":
        for a in range(M1 - 1):
            for b in range(M2):
                i1_list.append(a)
                j1_list.append(b)
                i2_list.append(a + 1)
                j2_list.append(b)
    elif monotonicity_dim1 == "decreasing":
        for a in range(M1 - 1):
            for b in range(M2):
                i1_list.append(a + 1)
                j1_list.append(b)
                i2_list.append(a)
                j2_list.append(b)

    if monotonicity_dim2 == "increasing":
        for a in range(M1):
            for b in range(M2 - 1):
                i1_list.append(a)
                j1_list.append(b)
                i2_list.append(a)
                j2_list.append(b + 1)
    elif monotonicity_dim2 == "decreasing":
        for a in range(M1):
            for b in range(M2 - 1):
                i1_list.append(a)
                j1_list.append(b + 1)
                i2_list.append(a)
                j2_list.append(b)

    if not i1_list:
        return params

    # Convert to tensors
    num_constraints = len(i1_list)
    i1 = torch.tensor(i1_list, dtype=torch.long, device=params.device)
    j1 = torch.tensor(j1_list, dtype=torch.long, device=params.device)
    i2 = torch.tensor(i2_list, dtype=torch.long, device=params.device)
    j2 = torch.tensor(j2_list, dtype=torch.long, device=params.device)
    
    # Dykstra's algorithm - properly tracking constraint residuals
    # Each increment tracks the difference added by a constraint projection
    increments = torch.zeros((num_constraints, 2), device=params.device, dtype=params.dtype)

    for _ in range(max_iter):
        x_old = x.clone()

        # Process each constraint sequentially
        for k in range(num_constraints):
            # Compute s = λ + p (add auxiliary variable to current parameters)
            s1 = x[i1[k], j1[k]] + increments[k, 0]
            s2 = x[i2[k], j2[k]] + increments[k, 1]
            
            # Project onto constraint: s2 >= s1
            if s2 < s1:
                mid = (s1 + s2) / 2
                r1, r2 = mid, mid
            else:
                r1, r2 = s1, s2
            
            # Update auxiliary variable: p_new = s - r
            increments[k, 0] = s1 - r1
            increments[k, 1] = s2 - r2
            
            # Update parameters: λ_new = r
            x[i1[k], j1[k]] = r1
            x[i2[k], j2[k]] = r2

        # Check convergence
        if torch.max(torch.abs(x - x_old)) < tol:
            break

    return x.view(-1)


def project_monotonic_1d(
    params: Tensor,
    monotonicity: str = "increasing",
) -> Tensor:
    """
    Project 1D parameters onto monotonic feasible set using pool adjacent
    violators algorithm (PAVA).

    Args:
        params: 1D tensor of parameters [M]
        monotonicity: "increasing" or "decreasing"

    Returns:
        Projected parameters [M]
    """
    if monotonicity == "none":
        return params

    x = params.clone()
    n = len(x)

    if monotonicity == "decreasing":
        x = -x

    # Pool Adjacent Violators Algorithm (PAVA)
    # More efficient than Dykstra for 1D case
    i = 0
    while i < n - 1:
        if x[i] > x[i + 1]:
            # Find the extent of the violation
            j = i + 1
            while j < n - 1 and x[j] > x[j + 1]:
                j += 1

            # Pool and average
            pool_sum = x[i:j + 1].sum()
            pool_avg = pool_sum / (j - i + 1)
            x[i:j + 1] = pool_avg

            # Check backwards
            while i > 0 and x[i - 1] > x[i]:
                i -= 1
                j_end = i + 1
                while j_end < n and x[j_end] == x[i + 1]:
                    j_end += 1
                pool_sum = x[i:j_end].sum()
                pool_avg = pool_sum / (j_end - i)
                x[i:j_end] = pool_avg
        else:
            i += 1

    if monotonicity == "decreasing":
        x = -x

    return x


class PWLCalibration(nn.Module):
    """
    Piecewise Linear (PWL) Calibration layer.

    Maps input values to [output_min, output_max] using piecewise linear
    interpolation, optionally with monotonicity constraints.

    This is used as the input layer for lattice regression to:
    1. Normalize inputs to the lattice domain [0, num_vertices-1]
    2. Enforce monotonicity on the mapping

    Args:
        input_min: Minimum input value
        input_max: Maximum input value
        num_keypoints: Number of keypoints for PWL function
        monotonicity: "increasing", "decreasing", or "none"
        output_min: Minimum output value (default 0.0)
        output_max: Maximum output value (default num_keypoints-1, but should
                    typically be set to num_vertices-1 to match the lattice)
    """

    def __init__(
        self,
        input_min: float,
        input_max: float,
        num_keypoints: int = 10,
        monotonicity: str = "none",
        output_min: float = 0.0,
        output_max: float = None,
    ):
        super().__init__()

        self.input_min = input_min
        self.input_max = input_max
        self.num_keypoints = num_keypoints
        self.monotonicity = monotonicity
        self.output_min = output_min
        self.output_max = output_max if output_max is not None else float(num_keypoints - 1)

        # Keypoint locations (fixed, uniformly spaced)
        keypoints = torch.linspace(input_min, input_max, num_keypoints)
        self.register_buffer("keypoints", keypoints)

        # Learnable output values at keypoints
        # Initialize linearly from output_min to output_max
        initial_outputs = torch.linspace(self.output_min, self.output_max, num_keypoints)
        self.keypoint_outputs = nn.Parameter(initial_outputs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply piecewise linear calibration.

        Args:
            x: Input tensor [batch_size] or [batch_size, 1]

        Returns:
            Calibrated output [batch_size]
        """
        if x.dim() == 2:
            x = x.squeeze(-1)

        # Clamp to valid range
        x = torch.clamp(x, self.input_min, self.input_max)

        # Find which segment each input falls into
        # Using searchsorted to find the right keypoint
        indices = torch.searchsorted(self.keypoints, x, right=True) - 1
        indices = torch.clamp(indices, 0, self.num_keypoints - 2)

        # Get keypoints for interpolation
        x_left = self.keypoints[indices]
        x_right = self.keypoints[indices + 1]
        y_left = self.keypoint_outputs[indices]
        y_right = self.keypoint_outputs[indices + 1]

        # Linear interpolation
        t = (x - x_left) / (x_right - x_left + 1e-8)
        output = y_left + t * (y_right - y_left)

        return output

    def project_monotonicity(self):
        """Project keypoint outputs to satisfy monotonicity constraints."""
        if self.monotonicity != "none":
            with torch.no_grad():
                projected = project_monotonic_1d(
                    self.keypoint_outputs.data,
                    self.monotonicity
                )
                self.keypoint_outputs.data.copy_(projected)


class Lattice(nn.Module):
    """
    Lattice layer for monotonic regression.

    Implements a 2D lookup table with bilinear interpolation.
    Monotonicity is enforced via Dykstra's projection after each
    optimization step.

    The lattice function is:
        f(x1, x2) = (1-t1)(1-t2)λ[a,b] + t1(1-t2)λ[a+1,b]
                  + (1-t1)t2λ[a,b+1] + t1*t2*λ[a+1,b+1]

    where t1, t2 are interpolation weights and λ are lattice parameters.

    Args:
        num_vertices_dim1: Number of vertices along dimension 1
        num_vertices_dim2: Number of vertices along dimension 2
        monotonicity_dim1: Monotonicity for dim 1 ("increasing", "decreasing", "none")
        monotonicity_dim2: Monotonicity for dim 2 ("increasing", "decreasing", "none")
    """

    def __init__(
        self,
        num_vertices_dim1: int,
        num_vertices_dim2: int,
        monotonicity_dim1: str = "none",
        monotonicity_dim2: str = "none",
    ):
        super().__init__()

        self.num_vertices_dim1 = num_vertices_dim1
        self.num_vertices_dim2 = num_vertices_dim2
        self.monotonicity_dim1 = monotonicity_dim1
        self.monotonicity_dim2 = monotonicity_dim2

        # Lattice parameters (learnable)
        # Initialize to zero
        self.lattice_params = nn.Parameter(
            torch.zeros(num_vertices_dim1 * num_vertices_dim2)
        )

    def forward(self, calibrated_inputs: list[Tensor]) -> Tensor:
        """
        Apply bilinear interpolation on the lattice.

        Args:
            calibrated_inputs: List of two tensors [x1, x2], each [batch_size]
                               Values should be in [0, num_vertices-1]

        Returns:
            Interpolated output [batch_size, 1]
        """
        x1, x2 = calibrated_inputs
        M1, M2 = self.num_vertices_dim1, self.num_vertices_dim2

        # Get lattice as 2D tensor
        lattice = self.lattice_params.view(M1, M2)

        # Clamp inputs to valid range
        x1 = torch.clamp(x1, 0, M1 - 1 - 1e-6)
        x2 = torch.clamp(x2, 0, M2 - 1 - 1e-6)

        # Get integer indices and fractional parts
        a = x1.long()
        b = x2.long()
        t1 = x1 - a.float()  # gamma_1 in paper
        t2 = x2 - b.float()  # gamma_2 in paper

        # Clamp indices to valid range
        a = torch.clamp(a, 0, M1 - 2)
        b = torch.clamp(b, 0, M2 - 2)

        # Get corner values
        v00 = lattice[a, b]
        v10 = lattice[a + 1, b]
        v01 = lattice[a, b + 1]
        v11 = lattice[a + 1, b + 1]

        # Bilinear interpolation
        output = (
            (1 - t1) * (1 - t2) * v00 +
            t1 * (1 - t2) * v10 +
            (1 - t1) * t2 * v01 +
            t1 * t2 * v11
        )

        return output.unsqueeze(-1)

    def project_monotonicity(self):
        """Project lattice parameters to satisfy monotonicity constraints."""
        if self.monotonicity_dim1 == "none" and self.monotonicity_dim2 == "none":
            return

        with torch.no_grad():
            projected = dykstra_projection(
                self.lattice_params.data,
                self.num_vertices_dim1,
                self.num_vertices_dim2,
                self.monotonicity_dim1,
                self.monotonicity_dim2,
            )
            self.lattice_params.data.copy_(projected)


class Lattice1D(nn.Module):
    """
    1D Lattice (linear spline) for monotonic univariate functions.

    Simpler version of Lattice for single-variable monotonic constraints.

    Args:
        num_vertices: Number of vertices (knots)
        monotonicity: "increasing", "decreasing", or "none"
    """

    def __init__(
        self,
        num_vertices: int,
        monotonicity: str = "none",
    ):
        super().__init__()

        self.num_vertices = num_vertices
        self.monotonicity = monotonicity

        # Lattice parameters
        self.lattice_params = nn.Parameter(torch.zeros(num_vertices))

    def forward(self, calibrated_input: Tensor) -> Tensor:
        """
        Apply linear interpolation.

        Args:
            calibrated_input: Calibrated input [batch_size], values in [0, num_vertices-1]

        Returns:
            Interpolated output [batch_size, 1]
        """
        x = calibrated_input
        M = self.num_vertices

        # Clamp to valid range
        x = torch.clamp(x, 0, M - 1 - 1e-6)

        # Get indices and interpolation weight
        a = x.long()
        t = x - a.float()

        a = torch.clamp(a, 0, M - 2)

        # Linear interpolation
        output = (1 - t) * self.lattice_params[a] + t * self.lattice_params[a + 1]

        return output.unsqueeze(-1)

    def project_monotonicity(self):
        """Project to satisfy monotonicity."""
        if self.monotonicity != "none":
            with torch.no_grad():
                projected = project_monotonic_1d(
                    self.lattice_params.data,
                    self.monotonicity
                )
                self.lattice_params.data.copy_(projected)


class MonotonicSubnet(nn.Module):
    """
    Complete monotonic subnetwork using PWL calibration + lattice.

    For main effects (1D) or interactions (2D) with monotonicity constraints.

    Args:
        input_bounds: Dict mapping feature name to (min, max) tuple
        monotonicity: Dict mapping feature name to monotonicity ("increasing"/"decreasing")
        num_keypoints: Number of PWL calibration keypoints
        num_vertices: Number of lattice vertices per dimension
    """

    def __init__(
        self,
        input_bounds: dict[str, tuple[float, float]],
        monotonicity: dict[str, str],
        num_keypoints: int = 20,
        num_vertices: int = 10,
    ):
        super().__init__()

        self.feature_names = list(input_bounds.keys())
        self.is_2d = len(self.feature_names) == 2
        self.num_vertices = num_vertices

        # Create calibrators for each input
        # output_max=num_vertices-1 ensures calibrator outputs match lattice input domain
        self.calibrators = nn.ModuleDict()
        for name in self.feature_names:
            min_val, max_val = input_bounds[name]
            mono = monotonicity.get(name, "none")
            self.calibrators[name] = PWLCalibration(
                min_val, max_val, num_keypoints, mono,
                output_min=0.0, output_max=float(num_vertices - 1)
            )

        # Create lattice
        if self.is_2d:
            name1, name2 = self.feature_names
            mono1 = monotonicity.get(name1, "none")
            mono2 = monotonicity.get(name2, "none")
            # For lattice, if calibrator is monotonic, lattice should be increasing
            # since calibrator handles the direction
            lattice_mono1 = "increasing" if mono1 != "none" else "none"
            lattice_mono2 = "increasing" if mono2 != "none" else "none"
            self.lattice = Lattice(
                num_vertices, num_vertices,
                lattice_mono1, lattice_mono2
            )
        else:
            name = self.feature_names[0]
            mono = monotonicity.get(name, "none")
            lattice_mono = "increasing" if mono != "none" else "none"
            self.lattice = Lattice1D(num_vertices, lattice_mono)

        # Centering layer for identifiability (subtracts mean, no variance scaling)
        self.output_center = RunningMeanCenter(1)
        
        # Output scale (applied after centering to control magnitude independently)
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """
        Forward pass.

        Args:
            inputs: Dict mapping feature name to tensor [batch_size] or [batch_size, 1]

        Returns:
            Output tensor [batch_size, 1]
        """
        # Calibrate inputs
        calibrated = []
        for name in self.feature_names:
            x = inputs[name]
            if x.dim() == 2:
                x = x.squeeze(-1)
            calibrated.append(self.calibrators[name](x))

        # Apply lattice
        if self.is_2d:
            output = self.lattice(calibrated)
        else:
            output = self.lattice(calibrated[0])

        # Center output for identifiability
        output = self.output_center(output)
        
        # Scale output (independent magnitude control)
        output = output * self.output_scale

        return output

    def project_monotonicity(self):
        """Project all parameters to satisfy monotonicity constraints."""
        for calibrator in self.calibrators.values():
            calibrator.project_monotonicity()
        self.lattice.project_monotonicity()
