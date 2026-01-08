import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from dynamic_lode.core.lode import LatentODE


@pytest.fixture
def model_config():
    return {
        "input_size": 3,
        "output_size": 3,
        "hidden_size": 10,
        "latent_size": 10,
        "width_size": 10,
        "depth": 2,
        "alpha": 0.1,
        "dt": 0.1,
        "key": jr.PRNGKey(0),
    }


def test_train_distance_loss(model_config):
    """Test the 'distance' loss path."""
    model = LatentODE(**model_config, lossType="distance")

    # Create dummy batch data
    ts = jnp.linspace(0, 1, 10)
    ys = jnp.zeros((10, 3))
    ts_ctx = jnp.linspace(0, 0.5, 5)
    ys_ctx = jnp.zeros((5, 3))
    latent_spread = jnp.ones(10)  # Dummy std dev

    loss = model.train(ts, ys, latent_spread, ts_ctx, ys_ctx, key=jr.PRNGKey(1))

    assert jnp.isfinite(loss)
    assert loss.shape == ()


def test_train_weighted_loss(model_config):
    """Test the 'weighted' loss path (different branch in code)."""
    model = LatentODE(**model_config, lossType="weighted")

    # Create dummy data with variance to test weighting
    ts = jnp.linspace(0, 1, 10)
    ys = jnp.array([jnp.sin(ts), jnp.cos(ts), ts]).T
    ts_ctx = jnp.linspace(0, 0.5, 5)
    ys_ctx = ys[:5]
    latent_spread = jnp.ones(10)

    loss = model.train(ts, ys, latent_spread, ts_ctx, ys_ctx, key=jr.PRNGKey(1))
    assert jnp.isfinite(loss)


def test_inference_methods(model_config):
    """Test public sampling methods not used in training."""
    model = LatentODE(**model_config, lossType="distance")
    ts = jnp.linspace(0, 1, 5)
    key = jr.PRNGKey(2)

    # Test sampleLatent (generative latent path)
    z_t = model.sampleLatent(ts, key=key)
    assert z_t.shape == (5, 10)  # (Time, Latent)

    # Test sample (generative data path)
    y_t = model.sample(ts, key=key)
    assert y_t.shape == (5, 3)  # (Time, Output)
