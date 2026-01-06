import jax.numpy as jnp
import jax
from jax.flatten_util import ravel_pytree

def top_hessian_eigenvalue(loss_fn, params, data, num_iters=50, key=jax.random.PRNGKey(0)):
    flat_params, unravel_fn = ravel_pytree(params)

    def flat_loss(flat_params):
        return loss_fn(unravel_fn(flat_params), data)

    def hvp(v):
        return jax.jvp(jax.grad(flat_loss), (flat_params,), (v,))[1]

    vec = jax.random.normal(key, shape=flat_params.shape, dtype=flat_params.dtype)
    vec = vec / (jnp.linalg.norm(vec) + 1e-8)

    for _ in range(num_iters):
        vec = hvp(vec)
        vec = vec / (jnp.linalg.norm(vec) + 1e-8)

    return jnp.dot(vec, hvp(vec))