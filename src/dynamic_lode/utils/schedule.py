from types import SimpleNamespace
import jax.numpy as jnp
import jax

# A container to hold the lr_array (initial value)
lr_holder = SimpleNamespace()
lr_holder.lr_array = jnp.ones(500) * 1e-3  # placeholder init


def lode_schedule(lr_holder, total_steps=10_000):
    """
    Creates a callable learning rate schedule backed by a mutable namespace.
    This implementation allows the schedule to be dynamically updated by modifying
    the `lr_holder` object externally. This is primarily useful for testing or
    interactive contexts where the schedule needs to change without recompiling
    the entire training graph.

    Args:
        lr_holder (SimpleNamespace): A mutable container with an attribute `lr_array`
                                     containing the schedule values.
        total_steps (int): The total number of training steps, used to interpolate
                           the fixed-size `lr_array` to the actual step count.

    Returns:
        callable: A function `fn(step) -> float` compatible with Optax.
    """

    def schedule_fn(step):
        lr_array = lr_holder.lr_array
        orig_steps = jnp.linspace(0, total_steps - 1, lr_array.shape[0])
        step = jnp.clip(step, 0, total_steps - 1)
        return jnp.interp(jnp.array([step]), orig_steps, lr_array)[0]

    return schedule_fn


# update lr without having to re-trace with JAX
def update_lr_buffer(buffer, new_array):
    """
    Updates the learning rate buffer in-place (conceptually) to avoid JIT recompilation.
    Because JAX JIT-compiles functions based on static array shapes, we cannot simply
    swap the schedule function. Instead, we maintain a fixed-size buffer and update
    its values.

    Args:
        buffer (jnp.ndarray): The existing LR schedule array (padded to full length).
        new_array (jnp.ndarray): The new predicted schedule segment.

    Returns:
        jnp.ndarray: The updated buffer with `new_array` overwriting the relevant section,
                     padded to match the original shape.
    """
    pad_len = buffer.shape[0] - new_array.shape[0]
    new_array = jnp.pad(new_array, (0, pad_len), constant_values=new_array[-1])
    return buffer.at[:].set(new_array)


# create an optax ready schedule function
def make_schedule_fn(lr_buffer, total_steps):
    """
    Creates an Optax-compatible schedule function backed by a JAX array buffer.
    This function defines the interpolation logic required to map the discrete
    learning rate predictions (from the LODE) to the continuous training steps.
    It handles clipping and linear interpolation to ensure smooth LR transitions.

    Args:
        lr_buffer (list): A mutable list containing the JAX array of learning rates
                          at index 0. This "box" pattern is used to pass the
                          current schedule into the closure.
        total_steps (int): The global step count for the entire training run,
                           defining the domain of the interpolation.

    Returns:
        callable: A function `fn(step) -> float` that returns the interpolated learning rate
                  for the given training step.
    """

    def schedule_fn(step):
        buffer = lr_buffer[0]  # must be list to update inside JIT
        x = jnp.linspace(0, total_steps, buffer.shape[0])
        step = jnp.clip(step, 0, total_steps - 1)
        return jnp.interp(jnp.array([step]), x, buffer)[0]

    return schedule_fn
