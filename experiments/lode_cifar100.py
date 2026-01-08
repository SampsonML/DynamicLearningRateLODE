# experimantal runscript for CIFAR100 dynamic LODE tests: https://arxiv.org/abs/2509.23052
import os
import math
import time
from functools import partial
from collections import defaultdict
from typing import Any, Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
from jax import config

# we use float64 for paper experiments, though this is not required
config.update("jax_enable_x64", True)
from flax import linen as nn
from flax.training import train_state, common_utils
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

tf.config.set_visible_devices([], "GPU")
jax.local_devices()

from dynamic_lode.core.lode import LatentODE
from dynamic_lode.core.lode_scheduler import lode_scheduler
from dynamic_lode.utils import update_lr_buffer, make_schedule_fn


# ---------------------
#    data processing
def augment_image(img):
    img = tf.image.resize_with_crop_or_pad(img, 40, 40)
    img = tf.image.random_crop(img, [32, 32, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    return img


def train_process_sample(x):
    image = augment_image(x["image"])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return {"image": image, "label": x["label"]}


def val_process_sample(x):
    image = tf.image.convert_image_dtype(x["image"], dtype=tf.float32)
    return {"image": image, "label": x["label"]}


def prepare_train_dataset(dataset_builder, batch_size, split="train"):
    ds = dataset_builder.as_dataset(split=split)
    ds = ds.repeat()
    ds = ds.map(train_process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    df = ds.shuffle(16 * batch_size, reshuffle_each_iteration=True, seed=0)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(10)
    return ds


def prepare_val_dataset(dataset_builder, batch_size, split="test"):
    ds = dataset_builder.as_dataset(split=split)
    ds = ds.map(val_process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(10)
    return ds


def tf_to_numpy(xs):
    return jax.tree_util.tree_map(lambda x: x._numpy(), xs)


def dataset_to_iterator(ds):
    it = map(tf_to_numpy, ds)
    return it


#   grab the model
from dynamic_lode.models.ResNet18 import ResNet


def cross_entropy_loss(logits, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    loss = jnp.mean(loss)
    return loss


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


# train state helper func
class TrainState(train_state.TrainState):
    batch_stats: Any


# function to get the schedule
def get_schedule(schedule_type, init_lr, total_steps):
    """
    Allows easier selection of optax learning rate schedules.
    Args:
        schedule_type (str): One of ['cosine', 'onecycle', 'flat', 'decay'].
        init_lr (float): The peak or initial learning rate.
        total_steps (int): Total number of training steps (decay horizon).

    Returns:
        optax.schedule: An Optax-compatible schedule function.

    Raises:
        ValueError: If an unsupported schedule type is provided.
    """
    if schedule_type == "cosine":
        return optax.cosine_decay_schedule(init_value=init_lr, decay_steps=total_steps)
    elif schedule_type == "onecycle":
        return optax.cosine_onecycle_schedule(
            transition_steps=total_steps,
            peak_value=init_lr,
        )
    elif schedule_type == "flat":
        return optax.constant_schedule(init_lr)
    elif schedule_type == "decay":
        return optax.exponential_decay(
            init_value=init_lr,
            transition_steps=max(total_steps // 3, 1),
            decay_rate=0.5,
            staircase=True,
        )
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")


# training step
@jax.jit
def train_step(state, batch, dropout_rng):
    """
    Performs a single training step including backprop and state updates.
    Args:
        state (TrainState): The current training state (params + optimizer).
        batch (dict): A batch of data {'image': ..., 'label': ...}.
        dropout_rng (jax.random.PRNGKey): Base RNG key for dropout.

    Returns:
        tuple: (updated_state, metrics_dict)
    """
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            batch["image"],
            train=True,
            rngs={"dropout": dropout_rng},
            mutable="batch_stats",
        )
        loss = cross_entropy_loss(logits, batch["label"])
        return loss, (new_model_state, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch["label"])
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )
    return new_state, metrics


# create the lode helper function
def calculate_lode_lr(
    lode,
    loss_trajectory,
    lr_trajectory,
    accuracy,
    extrap_len,
    step_current,
    best_loss,
    best_lr,
    best_test,
    lr_array,
    update_freq,
    t_final,
):
    """
    Prepares the context window and queries the LODE scheduler for the next LR schedule.
    This function slices the most recent `update_freq` steps
    from the history metrics to create the "context path" that the LODE encodes
    to determine the current training dynamics. In future this function will be
    removed and directly ingegrated into the lode_schedule.py pipeline

    Args:
        lode (LatentODE): The pre-trained Latent ODE model instance.
        loss_trajectory (list/array): Full history of training losses.
        lr_trajectory (list/array): Full history of learning rates (log-space).
        accuracy (list/array): Full history of validation accuracies.
        extrap_len (int): Total length of the extrapolation horizon
        step_current (int): The current global training step.
        best_loss, best_lr, best_test: Metadata for tracking best performance (unused in this wrapper).
        lr_array (jnp.ndarray): The current full-length learning rate schedule buffer.
        update_freq (int): The number of recent steps to use as the context window for the LODE.
        t_final (int): The target end step of the training run (prediction horizon).

    Returns:
        jnp.ndarray: An updated learning rate schedule array where the future values
                     have been replaced by the LODE's optimal prediction.
    """
    n = len(loss_trajectory)
    steps_n = update_freq
    time_path = jnp.arange(n - steps_n, n)
    steps_n = len(time_path)
    loss_path = loss_trajectory[-steps_n:]
    lr_path = lr_trajectory[-steps_n:]
    accuracy_path = accuracy[-steps_n:]
    reward = -1
    # run the lode scheduler
    lr_array = lode_scheduler(
        step_current,
        lode,
        time_path,
        loss_path,
        lr_path,
        accuracy_path,
        lr_array,
        t_final,
        reward_step=reward,
        sigma=0.15,
        loss_tol=2,
        n_samples=30,
        verbose=True,
    )

    return lr_array


# model eval
@jax.jit
def eval_step(state, batch):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    metrics = compute_metrics(logits, batch["label"])
    return metrics


def metrics_summary(metrics):
    metrics = jax.device_get(metrics)
    metrics = jax.tree_util.tree_map(lambda *args: np.stack(args), *metrics)
    summary = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
    return summary


def log_metrics(history, summary, name):
    print(f"{name}: ", end="", flush=True)
    for key, val in summary.items():
        history[name + " " + key].append(val)
        print(f"{key} {val:.3f} ", end="")


# define training loop
def train(state, train_iter, val_iter, test_iter, epochs, lr_array):
    """
    Main training loop with Dynamic LODE Interaction.
    Mechanism:
    - Runs standard training for `update_freq` steps.
    - Collects metrics (loss, accuracy, current LR) into a trajectory vector.
    - Calls `calculate_lode_lr` to extrapolate and optimize the future schedule.
    - Updates the optimizer's `lr_buffer` in-place to apply the new schedule
      without triggering a JIT re-compilation.
    - Supports on-the-fly switching from AdamW to Nesterov (via `switch_nest`)
      if the schedule demands it (experimental feature).

    Args:
        state (TrainState): Initial training state.
        train_iter, val_iter, test_iter: Data iterators.
        epochs (int): Total epochs to run.
        lr_array (jnp.ndarray): Initial buffer for the learning rate schedule.

    Returns:
        dict: A history dictionary containing loss, accuracy, and LR curves.
    """
    history = defaultdict(list)
    update_lr = False
    update_freq = 30
    virtual_step = 0
    update_step = 0
    cur_t = 0
    t_final = 600
    extrap_init = (
        t_final  # this can be altered depending on confidence of lode extrapolation
    )
    switch_nest = False
    best_lr = jnp.array(lr_array[cur_t])

    for epoch in range(1, epochs + 1):
        print(f"Epoch:{epoch}/{epochs} - ", end="")
        train_metrics = []
        log_every = train_steps_per_epoch // 9
        for step in range(train_steps_per_epoch):
            # update step
            if (step % log_every == 0) and train_metrics:
                cur_t += 1
                summary = metrics_summary(train_metrics)
                history["train lr"].append(float(schedule_fn(state.step)))
                log_metrics(history, summary, "train")
                print(f"train lr {np.array(history['train lr'])[-1]:.6f}")
                print("| ", end="")

                # perform the lr update
                virtual_step += 1
                # update routine
                update_step += 1
                if update_step > update_freq:
                    update_step = 0
                    update_lr = True

            batch = next(train_iter)
            state, metrics = train_step(state, batch, dropout_rng)
            train_metrics.append(metrics)

            if switch_nest:
                switch_nest = False
                new_tx = optax.adamw(
                    learning_rate=schedule_fn,
                    weight_decay=weight_decay,
                    nesterov=False,
                    b1=0.9,
                )
                state = state.replace(tx=new_tx)

        val_metrics = []
        log_every = val_steps_per_epoch // 9
        for step in range(val_steps_per_epoch):
            batch = next(val_iter)
            metrics = eval_step(state, batch)
            val_metrics.append(metrics)

            if step % log_every == 0:
                summary = metrics_summary(val_metrics)
                log_metrics(history, summary, "val")
                print("| ", end="")

        test_metrics = []
        log_every = test_steps_per_epoch // 9
        for step in range(test_steps_per_epoch):
            batch = next(test_iter)
            metrics = eval_step(state, batch)
            test_metrics.append(metrics)

            if step % log_every == 0:
                summary = metrics_summary(test_metrics)
                log_metrics(history, summary, "test")
                print()

        # perform lode lr update
        if update_lr == True:
            update_lr = False
            extrap_len = extrap_init + cur_t
            loss_vec = jnp.array(history["train loss"])
            val_acc_vec = jnp.array(history["val accuracy"])
            lr_vec = jnp.log(jnp.array(history["train lr"]))
            lr_array = calculate_lode_lr(
                lode,
                loss_vec,
                lr_vec,
                val_acc_vec,
                extrap_len,
                cur_t,
                best_loss,
                lr_vec[-1],
                best_test,
                lr_array,
                update_freq,
                t_final,
            )

            # update to the new learning rate schedule via buffer routine
            # this avoids a JIT recompile
            lr_buffer[0] = update_lr_buffer(lr_buffer[0], lr_array)
            new_tx = optax.adamw(
                learning_rate=schedule_fn,
                weight_decay=weight_decay,
                nesterov=False,
                b1=0.9,
            )
            step_ = state.step + 1
            state = state.replace(tx=new_tx)
            switch_nest = True

    return history


# parse in training parameters
parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--lr", type=float, default=5e-2, help="Initial learning rate")
parser.add_argument(
    "--schedule", type=str, default="onecycle", help="Learning rate schedule"
)
parser.add_argument(
    "--path", type=str, default="", help="path to the trained latentODE"
)
args = parser.parse_args()


# -------------------- #
#     get the lode     #
# -------------------- #
# lode hyperparams
# Note: these are hardcoded from experimental trials, a packaged train-script will remove these in future
hidden_size = 20  # hidden size of the RNN
latent_size = 20  # latent size of the autoencoder
width_size = 20  # width of the ODE
depth = 2  # depth of the ODE
alpha = 0.01  # strength of the path penalty
seed = 1992  # random seed
lossType = "distance"  # {default, mahalanobis, distance}
model_size = 3  # model has loss, lr, validation accuracy
key = jr.PRNGKey(seed)
data_key, model_key, loader_key, train_key, sample_key = jr.split(key, 5)
# instantiate the model
lode = LatentODE(
    input_size=model_size,
    output_size=model_size,
    hidden_size=hidden_size,
    latent_size=latent_size,
    width_size=width_size,
    depth=depth,
    key=model_key,
    alpha=alpha,
    lossType=lossType,
    dt=0.2,
)

# load the model
try:
    # Attempt to load local model if it exists
    lode = eqx.tree_deserialise_leaves(args.path, lode)
    print("Loaded pre-trained LODE model.")
except (FileNotFoundError, ValueError):
    print(
        "[WARNING] Pre-trained model not found. Using random initialization for demonstration."
    )
    # lode is initialized with random weights

# set fixed values for experimental runs
num_classes = 100
batch_size = 256
epochs = 60
weight_decay = 1e-2
# lode optimiser params
t_final = 600
best_loss = float("inf")
best_lr = args.lr
update_step = 0
virtual_step = 0
lr_array = np.ones(t_final) * args.lr
update_lr = False
best_test = 0
loss_track = []
lr_track = []
val_track = []

# load in the data locally from scratch (faster IO than home)
data_dir = os.environ.get("DATA_DIR", "./data")
dataset_builder = tfds.builder("cifar100", data_dir=data_dir)
dataset_builder.download_and_prepare()

# make train/validation/test splits
train_split = "train[:90%]"
val_split = "train[90%:]"
test_split = "test"
train_ds = prepare_train_dataset(dataset_builder, batch_size, split=train_split)
val_ds = prepare_val_dataset(dataset_builder, batch_size, split=val_split)
test_ds = prepare_val_dataset(dataset_builder, batch_size, split=test_split)
train_steps_per_epoch = math.ceil(
    dataset_builder.info.splits["train"].num_examples * 0.9 / batch_size
)
val_steps_per_epoch = math.ceil(
    dataset_builder.info.splits["train"].num_examples * 0.1 / batch_size
)
test_steps_per_epoch = math.ceil(
    dataset_builder.info.splits["test"].num_examples / batch_size
)

# create the iterators
train_iter = dataset_to_iterator(train_ds)
val_iter = dataset_to_iterator(val_ds)
test_iter = dataset_to_iterator(test_ds)

# take parsed values
learning_rate = args.lr
seed = args.seed
schedule = args.schedule
rng = jr.PRNGKey(seed)

# instantiate the model
model = ResNet(
    num_classes,
    channel_list=[64, 128, 256, 512],
    num_blocks_list=[2, 2, 2, 2],
    strides=[1, 1, 2, 2, 2],
    head_p_drop=0.3,
)


# instantiate model and initialize training state
@jax.jit
def initialize(params_rng):
    init_rngs = {"params": params_rng}
    input_shape = (1, 32, 32, 3)
    variables = model.init(init_rngs, jnp.ones(input_shape, jnp.float32), train=False)
    return variables


# make initial weights
params_rng, dropout_rng = jax.random.split(rng)
variables = initialize(params_rng)

# training optimizer setup
num_train_steps = train_steps_per_epoch * epochs
schedule_fn = get_schedule(schedule, learning_rate, num_train_steps)
tx = optax.adamw(learning_rate=schedule_fn, weight_decay=weight_decay, b1=0.9)
lr_array = jnp.array([schedule_fn(i) for i in jnp.arange(num_train_steps)])
by_idx = int(num_train_steps / t_final)
lr_array = lr_array[::by_idx]
best_validation = 0
init_lr = args.lr
lr_array = jnp.array(lr_array)[0:t_final]
# use buffer
lr_buffer = [lr_array]
schedule_fn = make_schedule_fn(lr_buffer, num_train_steps)
tx = optax.adamw(
    learning_rate=schedule_fn, weight_decay=weight_decay, nesterov=True, b1=0.9
)

from flax.struct import dataclass, field


@dataclass
class CustomTrainState(train_state.TrainState):
    batch_stats: Any = None
    # Use default_factory for the JAX array
    lr: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0, dtype=jnp.float32))


# create the initial training state
state = CustomTrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    batch_stats=variables["batch_stats"],
    tx=tx,
    lr=jnp.array(learning_rate, dtype=jnp.float32),
)

# train model and save metrics
history = train(state, train_iter, val_iter, test_iter, epochs, lr_array)

# save training metrics
import pickle

save_dir = ""  # change this to where you would like to save the traning metrics
save_name = f"schedule_{schedule}_lr_{learning_rate}_seed_{seed}.pkl"
with open(save_dir + save_name, "wb") as f:
    pickle.dump(history, f)

print(f"file saved to {save_dir}")
