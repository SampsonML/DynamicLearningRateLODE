# ---------------------------------- #
#     the lode scheduler routine     #
# ---------------------------------- #
import numpy as np
import jax
from jax import debug
import jax.numpy as jnp
from jax import random
import jax.random as jr
from jax import config
config.update("jax_enable_x64", True)
from .lode import LatentODE
from jax import vmap
from jax.typing import ArrayLike


def lode_scheduler(
    current_time: int,
    model: LatentODE,
    time_path: ArrayLike,
    loss_path: ArrayLike,
    lr_path: ArrayLike,
    validation_path: ArrayLike,
    lr_schedule: ArrayLike,
    t_final: int,
    reward_step: int =-1,
    sigma: float =0.15,
    loss_tol: float =2.0,
    n_samples: int =50,
    verbose: bool =True
) -> jnp.ndarray:
    """
    Performs probabilistic extrapolation of training loss curves in latent space using a trained latent ODE model.
    This function perturbs the current latent state of a training trajectory and decodes 
    multiple extrapolated future loss curves from sampled latent points. Each sample is 
    then accepted/rejected based on its similarity to reference performance metrics, and 
    the sample with the best projected reward outcome being selected. The function is primarily 
    intended for, dynamic learning rate adaptation, or analysis of training trajectory stability.

    Args:
        current_time (int): 
            The current training step or timestamp in the training trajectory.

        model (LatentODE): 
            A trained latent ODE model that provides methods for latent encoding and decoding.

        time_path (np.ndarray): 
            Array of shape (T,). Sequence of time indices corresponding to the observed training path.

        loss_path (np.ndarray): 
            Array of shape (T,). Sequence of training loss values up to the current time.

        lr_path (np.ndarray): 
            Array of shape (T,). Sequence of learning rates used during training.

        validation_path (np.ndarray): 
            Array of shape (T,). Sequence of observed validation loss.

        lr_schedule (np.ndarray): 
            The original or baseline learning rate schedule for reference. This is not used in scoring.

        t_final (int):
            The final time (in training steps) of the training routine

        sigma (float): 
            Standard deviation of Gaussian noise used to perturb the current latent state.

        loss_tol (int):
            A multiplier on how close the predicted loss must be to current loss for a valid path.
            The predicted loss must be within loss_tol * (std(loss_path)) of the old loss.

        verbose (bool, optional): 
            If True, prints additional information about the LODE prediction results. Defaults to True.

    Returns:
        lr_array (np.ndarray): 
            A forecasted learning rate schedule and accuracy trajectory corresponding to 
            the best-scoring sampled extrapolation.

    """

    # choose lode optimization hyperparameters
    prev_lr = jnp.exp(lr_path[-1])
    avg_steps = min(len(loss_path), 3) # early training metrics are volatile
    steps_left = t_final - current_time  
    if reward_step == -1: reward_step = steps_left
    if reward_step > steps_left:
        reward_step = steps_left
    extrap_path = jnp.arange(0, 2 * t_final) 
    full_path = jnp.array([loss_path, lr_path, validation_path]).T 

    # now determine the current latent vector (z_0) for the path taken, note treat lode_scheduler as a
    # private function in full LODE-scheduler pipeline, so calling _ methods here
    current_latent = model._latent(time_path, full_path, key=random.PRNGKey(23))
    current_traj = model._sample(extrap_path, current_latent)
    best_validation = validation_path[-1]

    # calculate average metrics 
    true_loss_avg = jnp.mean(loss_path[-avg_steps:])
    true_loss_std = jnp.std(loss_path[-avg_steps:])
    true_val_avg  = jnp.mean(validation_path[-avg_steps:])
    true_val_std = jnp.std(validation_path[-avg_steps:])

    # ----------------------------------------------------
    # step 1: sample 'n_samples' latent ODE extrapolations
    key = jax.random.PRNGKey(23)
    noise_matrix = jax.random.normal(key, shape=(n_samples, *current_latent.shape)) * sigma
    noise_matrix *= jnp.abs(current_latent)  # scale noise by magnitude

    # Include original latent as first sample
    noise_matrix = jnp.concatenate([jnp.zeros_like(current_latent)[None, :], noise_matrix], axis=0)
    noisy_latents = current_latent[None, :] + noise_matrix

    # Vectorized sampling
    batched_sample = jax.vmap(model._sample, in_axes=(None, 0))
    pred_trajs = batched_sample(extrap_path, noisy_latents)

    min_t = 0
    max_t = t_final

    # Preallocate arrays
    max_candidates = n_samples + 1
    closest_array = jnp.zeros((max_candidates, 4))  # stores [loss_diff, best_time, loss_final, val_final]
    latent_stack = jnp.zeros((max_candidates, *current_latent.shape))
    found_mask = jnp.zeros((max_candidates,), dtype=bool)
    insert_idx = 0

    # compute prediction arrays outside loop
    pred_lrs = jnp.exp(pred_trajs[:, :, 1]) 
    pred_loss = pred_trajs[:,:,0]
    pred_vals = pred_trajs[:,:,2]
    loss_diffs = jnp.abs(pred_trajs[:,:,0]) - true_loss_avg
    val_diffs = jnp.abs(pred_trajs[:,:,2]) - true_val_avg

    for sample in range(pred_trajs.shape[0]-1):
        pred_traj = pred_trajs[sample]
        pred_lr = pred_lrs[sample]
        loss_diff = loss_diffs[sample]
        val_diff = val_diffs[sample]
        # ---------------------------------
        # mask out the rejection conditions 
        loss_mask = loss_diff < (loss_tol * true_loss_std)
        time_mask = (jnp.arange(pred_traj.shape[0]) >= min_t) & (jnp.arange(pred_traj.shape[0]) < max_t)
        valid_mask = loss_mask & time_mask 

        if jnp.any(valid_mask):
            times = jnp.arange(pred_traj.shape[0])[valid_mask]
            best_idx = jnp.argmax(pred_traj[times,1]) # choose largest learning rate
            best_time_point = times[best_idx]
            reward_step_ = best_time_point + reward_step
            loss_final = pred_traj[reward_step_, 0]
            validation_final = pred_traj[reward_step_, 2]

            # Store into preallocated arrays
            closest_array = closest_array.at[insert_idx].set(
                jnp.array([pred_lr[best_time_point], best_time_point, loss_final, validation_final])
            )
            latent_stack = latent_stack.at[insert_idx].set(noisy_latents[sample])
            found_mask = found_mask.at[insert_idx].set(True)
            insert_idx += 1

    # Filter only valid rows
    valid_indices = jnp.where(found_mask)[0]
    closest_array = closest_array[valid_indices]
    latent_stack = latent_stack[valid_indices]
    pred_lrs = pred_lrs[valid_indices]
    pred_loss = pred_loss[valid_indices]
    pred_vals = pred_vals[valid_indices]

    if verbose: debug.print("\n----------- lode optimizer results -----------")

    # -------------------------------------------------
    # step 2.5: escape if no similar trajectories found return prev lr schedule
    #if not_found:
    if valid_indices.shape[0] == 0:
        if verbose:
            debug.print("     [!] all proposals rejected")
            debug.print("     [!] fallback to previous lr schedule")
            debug.print("----------------------------------------------\n")
        return lr_schedule
    if verbose: 
        debug.print("              ✓  new path found")
        debug.print("   ▸  samples accepted:           {}/{}",
                            closest_array.shape[0], n_samples)

    # -------------------------------------------------------------
    # step 3: choose the optimal new franken-curve and re-sample it
    reward_values = closest_array[:,3] 
    ensemble_n = jnp.minimum(3, reward_values.shape[0])
    best_indices = jnp.argsort(reward_values)[-ensemble_n:] # take the n largest values
    best_times = closest_array[best_indices, 1] 
    best_latents = latent_stack[best_indices] 
    best_validation = jnp.mean(closest_array[best_indices,3])
    lr_schedule_avg = jnp.zeros(steps_left)
    best_loss, best_val, best_lr = 0, 0, 0

    debug.print("   ▸  current time is:         {:.1f}", current_time)
    for i, t in zip(best_indices, best_times):
        lr = pred_lrs[i][int(t):int(t)+steps_left]
        lr_schedule_avg += lr
        best_loss += pred_loss[i][int(t)]
        best_val += pred_vals[i][int(t)]
        best_lr += pred_lrs[i][int(t)]
        debug.print("   ▸  sampled time is:         {:.1f}", t)

    lr_schedule_avg /= len(best_times)
    best_loss /= len(best_times)
    best_val /= len(best_times)
    best_lr /= len(best_times)
    eps = 1e-10
    loss_err = 100 * jnp.abs(best_loss - loss_path[-1]) / (loss_path[-1] + eps)
    lr_err = 100 * jnp.abs((best_lr - prev_lr) / (prev_lr + eps))
    val_err = 100 * jnp.abs(best_val - validation_path[-1]) / (validation_path[-1] + eps)

    if verbose:
        debug.print("   ▸  sampled loss error:         {:.3f}%",
                    loss_err)
        debug.print("   ▸  sampled lr error:           {:.3f}%",
                    lr_err)
        debug.print("   ▸  sampled accuracy error:     {:.3f}%",
                    val_err)
        debug.print("   ▸  updated lr:                 {:.3e}", lr_schedule_avg[0])
        debug.print("   ▸  previous lr:                {:.3e}", jnp.exp(lr_path[-1]))
        debug.print("----------------------------------------------\n")

    # pad the left of lr array with zeros ensuring it is always the same size
    # Note: the lode-scheduler never uses previously-seen values, so no need to store/concat 
    # the actual old learning rates (which may not not be the correct length)
    # a seperate array is saved to store full LR-schedule used in each run
    pad_len = t_final - steps_left
    pad = jnp.zeros(pad_len)
    lr_schedule_avg = jnp.concatenate([pad, lr_schedule_avg])

    return lr_schedule_avg

