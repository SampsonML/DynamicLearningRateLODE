import flax.linen as nn
import jax.numpy as jnp


class CNN(nn.Module):
    """
    A simple 3-layer Convolutional Neural Network baseline.
    Designed for Fashion-MNIST/MNIST tasks, this model includes hardcoded reshaping
    logic to handle flattened inputs if necessary. It uses a standard Conv-ReLU-Pool
    architecture with increasing channel depth and dropout regularization.

    Attributes:
        num_classes (int): Number of output classes (e.g., 10 for Fashion-MNIST).
        dropout_rate (float): Probability of dropping units during training. Default 0.25.
    """

    num_classes: int
    dropout_rate: float = 0.25

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.reshape((-1, 28, 28, 1))  # Updated for Fashion-MNIST input

        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        x = nn.Conv(128, (3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(128, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        x = nn.Dense(self.num_classes)(x)
        return x
