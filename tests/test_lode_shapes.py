"""
Tests for Latent ODE Architecture Dimensions.
This module verifies that the LatentODE model correctly processes input shapes
through the entire Encode-Process-Decode pipeline. It ensures that:
- The Encoder produces latent vectors of the correct size.
- The Decoder (ODE solver) returns trajectories matching the expected time and data dimensions.
"""

import pytest
import jax.random as jr
from dynamic_lode.core.lode import LatentODE


@pytest.fixture
def model_fixture():
    """Defines a standard model instance for testing."""
    return LatentODE(
        input_size=3,
        output_size=3,
        hidden_size=20,
        latent_size=20,
        width_size=20,
        depth=2,
        alpha=0.01,
        dt=0.1,
        key=jr.PRNGKey(0),
        lossType="default",
    )


def test_latent_ode_shapes(model_fixture):
    """
    Verifies the input-output tensor shapes of the Encoder and Decoder.
    Checks if:
    - _latent() maps (Time, Data) -> (Latent_Size,)
    - _sample() maps (Time,) + (Latent_Size,) -> (Time, Data)
    """
    import jax.numpy as jnp

    ts = jnp.linspace(0, 10, 50)
    ys = jnp.ones((50, 3))

    # Test encoding
    latent = model_fixture._latent(ts, ys, jr.PRNGKey(1))
    assert latent.shape == (20,)

    # Test decoding
    pred_ys = model_fixture._sample(ts, latent)
    assert pred_ys.shape == (50, 3)
