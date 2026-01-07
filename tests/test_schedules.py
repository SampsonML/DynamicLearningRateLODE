import jax.numpy as jnp
from dynamic_lode.utils.schedule import make_schedule_fn, update_lr_buffer


def test_schedule_interpolation():
    """
    Verifies that the schedule function correctly interpolates and handles
    indices out of bounds.
    """
    # Create a simple buffer: [0.0, 1.0, 2.0] mapped to 10 steps
    initial_buffer = jnp.array([0.0, 1.0, 2.0])
    # The box pattern used in your code
    lr_buffer = [initial_buffer]

    total_steps = 10
    schedule = make_schedule_fn(lr_buffer, total_steps)

    # Test start (step 0 matches index 0)
    assert jnp.isclose(schedule(0), 0.0)

    # Test middle (step 5 matches index 5 in linspace(0, 10, 3) -> [0, 5, 10])
    assert jnp.isclose(schedule(5), 1.0)

    # Test end (clipping behavior)
    # NOTE: The implementation clips step to (total_steps - 1), i.e., 9.
    # The domain is [0, 5, 10].
    # At step 9, we are 4/5ths of the way between 1.0 and 2.0.
    # Expected value = 1.0 + (0.8 * 1.0) = 1.8
    assert jnp.isclose(schedule(100), 1.8)


def test_buffer_update_padding():
    """
    Verifies that update_lr_buffer correctly pads a shorter new schedule
    to match the original buffer size.
    """
    # Original buffer size 10
    buffer = jnp.zeros(10)

    # New schedule prediction is shorter (e.g., remaining steps = 4)
    # [1, 1, 1, 1]
    new_schedule = jnp.ones(4)

    updated_buffer = update_lr_buffer(buffer, new_schedule)

    # Check shape preserved
    assert updated_buffer.shape == (10,)
    # Check values: first 4 should be 1.0, rest should be padded with the last value (1.0)
    assert jnp.all(updated_buffer == 1.0)
