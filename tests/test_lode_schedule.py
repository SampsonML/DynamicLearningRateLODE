import jax.numpy as jnp
import jax.random as jr
from dynamic_lode.core.lode_scheduler import lode_scheduler

# Define a mock LODE
class MockLatentODE:
    def _latent(self, ts, ys, key):
        # Return a dummy latent vector of size (20,)
        return jnp.ones(20)

    def _sample(self, ts, latent):
        # Return a dummy trajectory [loss, lr, val_acc]
        # Shape: (Time, 3)
        T = ts.shape[0]
        # Create fake curves:
        # Loss decreases (column 0)
        loss = jnp.linspace(1.0, 0.1, T)
        # LR is constant (column 1) - log space
        lr = jnp.zeros(T)
        # Val acc increases (column 2)
        val = jnp.linspace(0.5, 0.9, T)
        
        return jnp.stack([loss, lr, val], axis=-1)

def test_lode_scheduler_execution():
    """
    Verifies that lode_scheduler runs end-to-end with a mock model.
    Checks that it:
    - Accepts input paths.
    - Filters trajectories (mock always returns valid ones).
    - Returns a schedule of the correct length.
    """
    mock_model = MockLatentODE()
    
    # Dummy history data
    time_path = jnp.arange(10)
    loss_path = jnp.linspace(2.0, 1.0, 10)
    lr_path = jnp.zeros(10)
    val_path = jnp.linspace(0.2, 0.5, 10)
    current_lr_schedule = jnp.zeros(100) # Full schedule
    
    t_final = 50
    current_time = 10
    
    new_schedule = lode_scheduler(
        current_time=current_time,
        model=mock_model,
        time_path=time_path,
        loss_path=loss_path,
        lr_path=lr_path,
        validation_path=val_path,
        lr_schedule=current_lr_schedule,
        t_final=t_final,
        n_samples=5,      # Small number for speed
        verbose=False
    )
    
    # Output is an array
    assert isinstance(new_schedule, jnp.ndarray)
    
    # Output length matches the 'steps_left' + padding logic
    # The scheduler returns the full schedule length (t_final)
    # logic: pad_len + new_segment
    assert new_schedule.shape[0] == t_final
