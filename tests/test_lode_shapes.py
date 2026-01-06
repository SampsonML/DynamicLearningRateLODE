import pytest
import jax.random as jr
from dynamic_lode.core.lode import LatentODE

@pytest.fixture
def model_fixture():
    """Defines a standard model instance for testing."""
    return LatentODE(
        data_size=3,       # Matches loss, lr, val_acc
        hidden_size=20,
        latent_size=20,
        width_size=20,
        depth=2,
        alpha=0.01,
        key=jr.PRNGKey(0),
        lossType="default"
    )

def test_latent_ode_shapes(model_fixture):
    # now ensure currect latent dimensions
    import jax.numpy as jnp
    ts = jnp.linspace(0, 10, 50)
    ys = jnp.ones((50, 3))
    
    # Test encoding
    latent, mean, std = model_fixture._latent(ts, ys, jr.PRNGKey(1))
    assert latent.shape == (20,)
    
    # Test decoding
    pred_ys = model_fixture._sample(ts, latent)
    assert pred_ys.shape == (50, 3)
