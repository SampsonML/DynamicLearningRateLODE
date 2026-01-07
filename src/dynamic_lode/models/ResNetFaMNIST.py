# residual NNs in JAX (flax)
import flax.linen as nn
import jax.numpy as jnp


class ResidualBlock(nn.Module):
    filters: int
    strides: tuple = (1, 1)
    use_projection: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        if self.use_projection:
            residual = nn.Conv(self.filters, (1, 1), self.strides, use_bias=False)(x)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        x = nn.Conv(self.filters, (3, 3), self.strides, padding="SAME", use_bias=False)(
            x
        )
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(self.filters, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        return nn.relu(x + residual)


class ResNet18(nn.Module):
    """
    ResNet-18 variant optimized for Fashion-MNIST (28x28 input).
    Attributes:
        num_classes (int): Number of output classes. Default 10.
    """

    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Fashion MNIST: (batch, 28, 28, 1)
        x = nn.Conv(64, (3, 3), strides=(1, 1), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        def make_layer(filters, blocks, stride):
            layers = []
            layers.append(
                ResidualBlock(filters, strides=(stride, stride), use_projection=True)
            )
            for _ in range(1, blocks):
                layers.append(ResidualBlock(filters))
            return layers

        # ResNet-18 block configuration: [2, 2, 2, 2]
        for block in make_layer(64, 2, stride=1):
            x = block(x, train)
        for block in make_layer(128, 2, stride=2):
            x = block(x, train)
        for block in make_layer(256, 2, stride=2):
            x = block(x, train)
        for block in make_layer(512, 2, stride=2):
            x = block(x, train)

        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(self.num_classes)(x)
        return x


class ResNet9(nn.Module):
    """
    A lightweight 'ResNet-9' architecture for fast experimentation.
    Attributes:
        num_classes (int): Number of output classes. Default 10.
    """

    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(128, (3, 3), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = ResidualBlock(128)(x, train=train)

        x = nn.Conv(256, (3, 3), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(512, (3, 3), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = ResidualBlock(512)(x, train=train)

        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(self.num_classes)(x)
        return x
