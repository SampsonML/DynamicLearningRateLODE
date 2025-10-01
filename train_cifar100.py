# from: https://juliusruseckas.github.io/ml/flax-cifar10.html
import math
from functools import partial
from collections import defaultdict
from typing import Any, Sequence
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, common_utils
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
tf.config.set_visible_devices([], 'GPU')
jax.local_devices()


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
    image = augment_image(x['image'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return {'image': image, 'label': x['label']}

def val_process_sample(x):
    image = tf.image.convert_image_dtype(x['image'], dtype=tf.float32)
    return {'image': image, 'label': x['label']}

def prepare_train_dataset(dataset_builder, batch_size, split='train'):
    ds = dataset_builder.as_dataset(split=split)
    ds = ds.repeat()
    ds = ds.map(train_process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    df = ds.shuffle(16 * batch_size, reshuffle_each_iteration=True, seed=0)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(10)
    return ds

def prepare_val_dataset(dataset_builder, batch_size, split='test'):
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
from ResNet18 import ResNet

def cross_entropy_loss(logits, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    loss = jnp.mean(loss)
    return loss

def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
      'loss': loss,
      'accuracy': accuracy,
    }
    return metrics

# train state helper func
class TrainState(train_state.TrainState):
    batch_stats: Any

# function to get the schedule
def get_schedule(schedule_type, init_lr, total_steps):
    if schedule_type == 'cosine':
        return optax.cosine_decay_schedule(
            init_value=init_lr,
            decay_steps=total_steps
        )
    elif schedule_type == 'onecycle':
        return optax.cosine_onecycle_schedule(
            transition_steps=total_steps,
            peak_value=init_lr,
        )
    elif schedule_type == 'flat':
        return optax.constant_schedule(init_lr)
    elif schedule_type == 'decay':
        return optax.exponential_decay(
            init_value=init_lr,
            transition_steps=max(total_steps // 3, 1),
            decay_rate=0.5,
            staircase=True
        )
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")

# training step
@jax.jit
def train_step(state, batch, dropout_rng):
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)
    
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(variables, batch['image'], train=True,
                                                 rngs={'dropout': dropout_rng}, mutable='batch_stats')
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, (new_model_state, logits)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch['label'])
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    # add lr to metrics 
    lr = schedule_fn(state.step)
    metrics['lr'] = lr
    return new_state, metrics

# model eval
@jax.jit
def eval_step(state, batch):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
    metrics = compute_metrics(logits, batch['label'])
    return metrics

def metrics_summary(metrics):
    metrics = jax.device_get(metrics)
    metrics = jax.tree_util.tree_map(lambda *args: np.stack(args), *metrics)
    summary = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
    return summary

def log_metrics(history, summary, name):
    print(f"{name}: ", end='', flush=True)
    for key, val in summary.items():
        history[name + ' ' + key].append(val)
        print(f"{key} {val:.3f} ", end='')


# define training loop
def train(state, train_iter, val_iter, test_iter, epochs):
    history = defaultdict(list)
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch:{epoch}/{epochs} - ", end='')
        
        train_metrics = []
        log_every = train_steps_per_epoch // 9
        for step in range(train_steps_per_epoch):
            batch = next(train_iter)
            state, metrics = train_step(state, batch, dropout_rng)
            train_metrics.append(metrics)
        
            if step % log_every == 0:
                summary = metrics_summary(train_metrics)
                log_metrics(history, summary, 'train')
                print('| ', end='')
        
        val_metrics = []
        log_every = val_steps_per_epoch // 9
        for step in range(val_steps_per_epoch):
            batch = next(val_iter)
            metrics = eval_step(state, batch)
            val_metrics.append(metrics)
        
            if step % log_every == 0:
                summary = metrics_summary(val_metrics)
                log_metrics(history, summary, 'val')
                print('| ', end='')

        test_metrics = []
        log_every = test_steps_per_epoch // 9
        for step in range(test_steps_per_epoch):
            batch = next(test_iter)
            metrics = eval_step(state, batch)
            test_metrics.append(metrics)
            
            if step % log_every == 0:
                summary = metrics_summary(test_metrics)
                log_metrics(history, summary, 'test')
                print()
    
    return history

# parse in training parameters
parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--lr", type=float, default=5e-2, help="Initial learning rate")
parser.add_argument("--schedule", type=str, default="cosine", help="Learning rate schedule")
args = parser.parse_args()

# set fixed values
num_classes = 100
batch_size = 256
epochs = 60
weight_decay = 1e-2

# load in the data locally from scratch (faster IO than home)
data_dir = "/scratch/gpfs/ms0821/tfds_data/"
dataset_builder = tfds.builder('cifar100', data_dir=data_dir)
dataset_builder.download_and_prepare()

# make train/validation/test splits
train_split = 'train[:90%]'
val_split = 'train[90%:]'
test_split = 'test'
train_ds = prepare_train_dataset(dataset_builder, batch_size, split=train_split)
val_ds   = prepare_val_dataset(dataset_builder, batch_size, split=val_split)
test_ds  = prepare_val_dataset(dataset_builder, batch_size, split=test_split)
train_steps_per_epoch = math.ceil(dataset_builder.info.splits['train'].num_examples * 0.9 / batch_size)
val_steps_per_epoch = math.ceil(dataset_builder.info.splits['train'].num_examples * 0.1 / batch_size)
test_steps_per_epoch = math.ceil(dataset_builder.info.splits['test'].num_examples / batch_size)

# create the iterators
train_iter = dataset_to_iterator(train_ds)
val_iter = dataset_to_iterator(val_ds)
test_iter  = dataset_to_iterator(test_ds)

# take parsed values
learning_rate = args.lr
seed = args.seed
schedule = args.schedule
rng = jax.random.PRNGKey(seed)

# instantiate the model
model = ResNet(num_classes,
               channel_list = [64, 128, 256, 512],
               num_blocks_list = [2, 2, 2, 2],
               strides = [1, 1, 2, 2, 2],
               head_p_drop = 0.3)

# instantiate model and initialize training state
@jax.jit
def initialize(params_rng):
    init_rngs = {'params': params_rng}
    input_shape = (1, 32, 32, 3)
    variables = model.init(init_rngs, jnp.ones(input_shape, jnp.float32), train=False)
    return variables

# make initial weights
params_rng, dropout_rng = jax.random.split(rng) 
variables = initialize(params_rng) 

# training optimizer setup 
num_train_steps = train_steps_per_epoch * epochs
schedule_fn = get_schedule(schedule, learning_rate, num_train_steps)
tx = optax.adamw(learning_rate=schedule_fn, weight_decay=weight_decay)

# create the initial training state
state = TrainState.create(
    apply_fn = model.apply,
    params = variables['params'],
    batch_stats = variables['batch_stats'],
    tx = tx)

# train model and save metrics
history = train(state, 
                train_iter, 
                val_iter, 
                test_iter, 
                epochs)

print('training complete')

# save training metrics 
import pickle 
save_dir = "/scratch/gpfs/ms0821/lode_optimisers/latent-ode-optimizer/resnet_trials/cifar100/"
save_name = f"schedule_{schedule}_lr_{learning_rate}_seed_{seed}.pkl"
with open(save_dir+save_name, "wb") as f:
    pickle.dump(history, f)

print(f'file saved to {save_dir}')


