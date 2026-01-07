import jax.numpy as jnp
import jax
from jax.flatten_util import ravel_pytree


def top_hessian_eigenvalue(
    loss_fn, params, data, num_iters=50, key=jax.random.PRNGKey(0)
):
    """
    Estimates the top eigenvalue of the Hessian matrix using Power Iteration.
    This acts as a proxy for "sharpness" of the loss landscape. A higher max eigenvalue
    indicates a sharper minimum (higher curvature), which often correlates with
    poorer generalization.

    Args:
        loss_fn (callable): The loss function L(params, data).
        params (pytree): The current model parameters (weights).
        data (batch): A batch of data to evaluate curvature on.
        num_iters (int): Number of power iteration steps. More steps = higher precision.
        key (jax.random.PRNGKey): Random key for initializing the probe vector.

    Returns:
        float: The estimated maximum eigenvalue (λ_max) of the Hessian.
    """
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
