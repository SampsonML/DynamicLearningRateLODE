import flax.linen as nn
import jax.numpy as jnp


class CNN(nn.Module):
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



