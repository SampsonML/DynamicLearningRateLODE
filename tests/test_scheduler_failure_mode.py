import jax.numpy as jnp
from dynamic_lode.core.lode_scheduler import lode_scheduler


class MockBadODE:
    """A mock model that always predicts terrible losses."""

    def _latent(self, ts, ys, key):
        return jnp.ones(10)

    def _sample(self, ts, latent):
        T = ts.shape[0]
        # Return exploding loss (1000.0) so it fails the 'loss_tol' check
        loss = jnp.ones(T) * 1000.0
        lr = jnp.zeros(T)
        val = jnp.zeros(T)
        return jnp.stack([loss, lr, val], axis=-1)


def test_scheduler_fallback():
    """Ensure scheduler returns original schedule when all paths are rejected."""
    mock_model = MockBadODE()

    # Setup inputs
    t_final = 20
    current_time = 5
    old_schedule = jnp.ones(20) * 0.01  # The fallback

    # Use strict tolerance so our bad mock definitely fails
    new_schedule = lode_scheduler(
        current_time=current_time,
        model=mock_model,
        time_path=jnp.arange(5),
        loss_path=jnp.zeros(5),  # Current loss is 0
        lr_path=jnp.zeros(5),
        validation_path=jnp.zeros(5),
        lr_schedule=old_schedule,
        t_final=t_final,
        loss_tol=0.1,  # Strict tolerance
        n_samples=5,
        verbose=True,
    )

    # Should return the exact original schedule object/values
    assert jnp.array_equal(new_schedule, old_schedule)
