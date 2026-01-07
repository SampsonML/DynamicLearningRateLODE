"""
Tests for Mathematical Utility Functions.
This module validates the correctness of the custom JAX-based mathematical tools
used in the project, specifically the Hessian spectral analysis.
"""

import jax
import jax.numpy as jnp
from dynamic_lode.utils.hessian import top_hessian_eigenvalue


def test_hessian_calculation():
    """
    Verifies the Power Iteration algorithm on a known quadratic landscape.
    We define a dummy loss L(x) = 0.5 * x^T A x where A = diag(10, 1).
    Analytically, the Hessian is A, and its top eigenvalue is 10.0.
    This test asserts that `top_hessian_eigenvalue` recovers this value.
    """

    # Simple quadratic function: f(x) = 0.5 * x^T A x where A is diag(10, 1)
    def dummy_loss(params, data):
        A = jnp.array([[10.0, 0.0], [0.0, 1.0]])
        return 0.5 * jnp.dot(params, jnp.dot(A, params))

    params = jnp.array([1.0, 1.0])
    top_ev = top_hessian_eigenvalue(dummy_loss, params, data=None, num_iters=100)

    assert jnp.isclose(top_ev, 10.0, rtol=1e-2)
