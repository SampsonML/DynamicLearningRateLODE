import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from dynamic_lode.models.CNN import CNN
from dynamic_lode.models.ResNet34 import ResNet34, ResNet50
from dynamic_lode.models.ResNetFaMNIST import ResNet18 as ResNet18Fa, ResNet9


def run_forward_pass(model_class, input_shape, **kwargs):
    """Helper to initialize and run a model."""
    key = jr.PRNGKey(0)
    # Create dummy input
    x = jnp.ones((2, *input_shape))  # Batch size 2

    # Initialize
    model = model_class(**kwargs)
    variables = model.init(key, x, train=False)

    # 1. Test Eval Mode
    out = model.apply(variables, x, train=False)
    assert out.shape[0] == 2
    assert out.shape[1] == model.num_classes

    # 2. Test Train Mode (Dropout/BatchNorm)
    # Train mode requires a dropout key and mutable batch stats
    out_train = model.apply(
        variables, x, train=True, rngs={"dropout": key}, mutable=["batch_stats"]
    )
    # out_train is (output, new_state)
    assert out_train[0].shape == (2, model.num_classes)


def test_cnn_mnist():
    """Test the simple CNN baseline (expects flat or 28x28)."""
    # The CNN reshapes internally, so we can pass 28x28x1
    run_forward_pass(CNN, (28, 28, 1), num_classes=10)


def test_resnet34_imagenet():
    """Test Standard ResNet34 (usually for larger images)."""
    # Using 64x64 to speed up test, ResNet adapts to spatial dim
    run_forward_pass(ResNet34, (64, 64, 3), num_classes=100)


def test_resnet50_imagenet():
    """Test Standard ResNet50 (Bottleneck blocks)."""
    run_forward_pass(ResNet50, (64, 64, 3), num_classes=100)


def test_resnet18_famnist():
    """Test the Fashion-MNIST optimized ResNet18."""
    # Specifically expects 28x28x1 inputs
    run_forward_pass(ResNet18Fa, (28, 28, 1), num_classes=10)


def test_resnet9_famnist():
    """Test the lightweight ResNet9."""
    run_forward_pass(ResNet9, (28, 28, 1), num_classes=10)
