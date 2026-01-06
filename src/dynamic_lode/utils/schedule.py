from types import SimpleNamespace
import jax.numpy as jnp
import jax

# A container to hold the lr_array (initial value)
lr_holder = SimpleNamespace()
lr_holder.lr_array = jnp.ones(500) * 1e-3  # placeholder init

def lode_schedule(lr_holder, total_steps=10_000):
    def schedule_fn(step):
        lr_array = lr_holder.lr_array
        orig_steps = jnp.linspace(0, total_steps - 1, lr_array.shape[0])
        step = jnp.clip(step, 0, total_steps - 1)
        return jnp.interp(jnp.array([step]), orig_steps, lr_array)[0]
    return schedule_fn

# update lr without having to re-trace with JAX
def update_lr_buffer(buffer, new_array):
    pad_len = buffer.shape[0] - new_array.shape[0]
    new_array = jnp.pad(new_array, (0, pad_len), constant_values=new_array[-1])
    return buffer.at[:].set(new_array)

# create an optax ready schedule function
def make_schedule_fn(lr_buffer, total_steps):
    def schedule_fn(step):
        buffer = lr_buffer[0]  # must be list to update inside JIT
        x = jnp.linspace(0, total_steps, buffer.shape[0])
        step = jnp.clip(step, 0, total_steps - 1)
        return jnp.interp(jnp.array([step]), x, buffer)[0]
    return schedule_fn



