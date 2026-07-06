"""
Tests for Dykstra projection algorithm.
"""

import pytest
import torch
from anam.lattice import dykstra_projection


class TestDykstraProjection:
    """Test Dykstra's projection algorithm for monotonicity constraints."""
    
    def test_no_constraints_returns_unchanged(self):
        """When no constraints, should return original parameters."""
        params = torch.randn(16)
        result = dykstra_projection(params, 4, 4, "none", "none")
        torch.testing.assert_close(result, params)
    
    def test_increasing_dim1_enforced(self):
        """Test that dim1 increasing constraint is enforced."""
        # Create non-monotonic data: [3, 1, 2, 0] for each column
        params = torch.tensor([3.0, 1.0, 2.0, 0.0] * 4)
        
        result = dykstra_projection(
            params, 4, 4,
            monotonicity_dim1="increasing",
            monotonicity_dim2="none"
        )
        
        result_2d = result.view(4, 4)
        # Check each column is monotonically increasing
        for col in range(4):
            column = result_2d[:, col]
            for i in range(len(column) - 1):
                assert column[i] <= column[i+1], f"Column {col} not increasing at position {i}"
    
    def test_decreasing_dim1_enforced(self):
        """Test that dim1 decreasing constraint is enforced."""
        params = torch.tensor([0.0, 2.0, 1.0, 3.0] * 4)
        
        result = dykstra_projection(
            params, 4, 4,
            monotonicity_dim1="decreasing",
            monotonicity_dim2="none"
        )
        
        result_2d = result.view(4, 4)
        # Check each column is monotonically decreasing
        for col in range(4):
            column = result_2d[:, col]
            for i in range(len(column) - 1):
                assert column[i] >= column[i+1], f"Column {col} not decreasing at position {i}"
    
    def test_increasing_dim2_enforced(self):
        """Test that dim2 increasing constraint is enforced."""
        params = torch.tensor([3.0, 1.0, 2.0, 0.0,
                              3.0, 1.0, 2.0, 0.0,
                              3.0, 1.0, 2.0, 0.0,
                              3.0, 1.0, 2.0, 0.0])
        
        result = dykstra_projection(
            params, 4, 4,
            monotonicity_dim1="none",
            monotonicity_dim2="increasing"
        )
        
        result_2d = result.view(4, 4)
        # Check each row is monotonically increasing (with tolerance for float precision)
        tol = 1e-6
        for row in range(4):
            row_data = result_2d[row, :]
            for i in range(len(row_data) - 1):
                assert row_data[i] <= row_data[i+1] + tol, f"Row {row} not increasing at position {i}"
    
    def test_2d_monotonic_both_increasing(self):
        """Test 2D monotonicity with both dimensions increasing."""
        torch.manual_seed(42)
        params = torch.randn(64)
        
        result = dykstra_projection(
            params, 8, 8,
            monotonicity_dim1="increasing",
            monotonicity_dim2="increasing"
        )
        
        result_2d = result.view(8, 8)
        
        # Allow small violations due to finite iterations (100 is default)
        tol = 2e-4
        # Check dim1 (rows) increasing
        for col in range(8):
            for row in range(7):
                assert result_2d[row, col] <= result_2d[row+1, col] + tol
        
        # Check dim2 (cols) increasing  
        for row in range(8):
            for col in range(7):
                assert result_2d[row, col] <= result_2d[row, col+1] + tol
    
    def test_already_monotonic_unchanged(self):
        """If already monotonic, should remain very close to original."""
        # Create already monotonic data
        params = torch.arange(16, dtype=torch.float32)
        
        result = dykstra_projection(
            params, 4, 4,
            monotonicity_dim1="increasing",
            monotonicity_dim2="increasing"
        )
        
        # Should be very close to original
        torch.testing.assert_close(result, params, rtol=1e-5, atol=1e-5)
    
    def test_mixed_monotonicity(self):
        """Test with decreasing dim1 and increasing dim2."""
        torch.manual_seed(42)
        params = torch.randn(36)
        
        result = dykstra_projection(
            params, 6, 6,
            monotonicity_dim1="decreasing",
            monotonicity_dim2="increasing"
        )
        
        result_2d = result.view(6, 6)
        
        # Allow small violations due to finite iterations
        tol = 2e-4
        # Check dim1 decreasing
        for col in range(6):
            for row in range(5):
                assert result_2d[row, col] >= result_2d[row+1, col] - tol
        
        # Check dim2 increasing
        for row in range(6):
            for col in range(5):
                assert result_2d[row, col] <= result_2d[row, col+1] + tol
    
    def test_convergence_with_tolerance(self):
        """Test that algorithm converges within tolerance."""
        torch.manual_seed(42)
        params = torch.randn(16)
        
        result1 = dykstra_projection(
            params.clone(), 4, 4,
            monotonicity_dim1="increasing",
            monotonicity_dim2="increasing",
            max_iter=100,
            tol=1e-6
        )
        
        result2 = dykstra_projection(
            params.clone(), 4, 4,
            monotonicity_dim1="increasing",
            monotonicity_dim2="increasing",
            max_iter=1000,
            tol=1e-6
        )
        
        # With same tolerance, results should be very similar
        torch.testing.assert_close(result1, result2, rtol=1e-5, atol=1e-5)
    
    def test_device_preservation(self):
        """Test that result is on same device as input."""
        params = torch.randn(16)
        result = dykstra_projection(params, 4, 4, "increasing", "none")
        assert result.device == params.device
    
    def test_dtype_preservation(self):
        """Test that result has same dtype as input."""
        params = torch.randn(16, dtype=torch.float32)
        result = dykstra_projection(params, 4, 4, "increasing", "none")
        assert result.dtype == params.dtype
