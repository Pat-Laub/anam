"""
Loss functions for ANAM.

Includes:
- Gamma negative log-likelihood loss
- Smoothness penalty (2nd derivative regularization)
- Marginal clarity penalty (orthogonality constraint)
"""

import torch
from torch import Tensor


def gamma_nll(y_true: Tensor, y_pred: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Gamma negative log-likelihood loss (deviance form).

    This is the unit deviance for Gamma distribution:
        D(y, mu) = y/mu - log(y/mu) - 1

    We use the simplified form (dropping the constant -1):
        loss = mean(y/mu - log(y/mu))

    Args:
        y_true: Ground truth values (must be positive)
        y_pred: Predicted values (must be positive)
        eps: Small constant for numerical stability

    Returns:
        Scalar loss value
    """
    y_pred = torch.clamp(y_pred, min=eps)
    ratio = y_true / y_pred
    loss = torch.mean(ratio - torch.log(ratio))
    return loss


def smoothness_penalty(
    shape_outputs: Tensor,
    grid_spacing: float,
    lambda_smooth: float = 1e-4,
) -> Tensor:
    """
    Compute smoothness penalty using finite difference approximation
    of the second derivative.

    The penalty is: lambda * mean(|f''(x)|)

    where f''(x) is approximated by:
        f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2

    Args:
        shape_outputs: Shape function outputs evaluated on a grid [N]
        grid_spacing: Spacing between grid points (h)
        lambda_smooth: Regularization weight

    Returns:
        Scalar penalty value
    """
    if shape_outputs.shape[0] < 3:
        return torch.tensor(0.0, device=shape_outputs.device)

    h_squared = grid_spacing ** 2

    # Second derivative via central finite differences
    f_plus = shape_outputs[2:]      # f(x + h)
    f_current = shape_outputs[1:-1]  # f(x)
    f_minus = shape_outputs[:-2]     # f(x - h)

    second_deriv = (f_plus - 2 * f_current + f_minus) / h_squared
    penalty = lambda_smooth * torch.mean(torch.abs(second_deriv))

    return penalty


def marginal_clarity_penalty(
    main_outputs: dict[str, Tensor],
    interaction_outputs: dict[tuple[str, str], Tensor],
    lambda_mc: float = 1e-4,
) -> Tensor:
    """
    Compute marginal clarity penalty to ensure orthogonality between
    main effects and their associated interaction effects.

    The penalty is:
        P_mc = lambda * sum_{i,j} |mean(s_i(x_i) * h_ij(x_i, x_j))|

    This ensures main effects are orthogonal to interaction effects,
    avoiding confounding between them.

    Args:
        main_outputs: Dict mapping feature name -> shape function outputs [N]
        interaction_outputs: Dict mapping (var1, var2) -> interaction outputs [N]
        lambda_mc: Regularization weight

    Returns:
        Scalar penalty value
    """
    penalty = torch.tensor(0.0)
    device = None

    for (var1, var2), int_output in interaction_outputs.items():
        if device is None:
            device = int_output.device
            penalty = penalty.to(device)

        # Check orthogonality with both main effects
        if var1 in main_outputs:
            main1 = main_outputs[var1]
            penalty = penalty + torch.abs(torch.mean(main1 * int_output))

        if var2 in main_outputs:
            main2 = main_outputs[var2]
            penalty = penalty + torch.abs(torch.mean(main2 * int_output))

    return lambda_mc * penalty


def poisson_nll(y_true: Tensor, y_pred: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Poisson negative log-likelihood loss.

    loss = mean(y_pred - y_true * log(y_pred))

    Args:
        y_true: Ground truth counts (non-negative integers)
        y_pred: Predicted rate (must be positive)
        eps: Small constant for numerical stability

    Returns:
        Scalar loss value
    """
    y_pred = torch.clamp(y_pred, min=eps)
    loss = torch.mean(y_pred - y_true * torch.log(y_pred))
    return loss
