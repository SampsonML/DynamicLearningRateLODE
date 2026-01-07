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


class ResNet34(nn.Module):
    """
    Standard ResNet-34 implementation using Basic Residual Blocks.
    Follows the classic [3, 4, 6, 3] block structure with filter counts [64, 128, 256, 512].
    Suitable for larger scale image classification tasks.

    Attributes:
        num_classes (int): Number of output classes. Default 1000 (ImageNet).
    """

    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(64, (7, 7), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        def make_layer(filters, blocks, stride):
            layers = []
            layers.append(
                ResidualBlock(filters, strides=(stride, stride), use_projection=True)
            )
            for _ in range(1, blocks):
                layers.append(ResidualBlock(filters))
            return layers

        # ResNet-34 block configuration
        for block in make_layer(64, 3, stride=1):
            x = block(x, train)
        for block in make_layer(128, 4, stride=2):
            x = block(x, train)
        for block in make_layer(256, 6, stride=2):
            x = block(x, train)
        for block in make_layer(512, 3, stride=2):
            x = block(x, train)

        x = jnp.mean(x, axis=(1, 2))  # global average pooling
        x = nn.Dense(self.num_classes)(x)
        return x


class BottleneckBlock(nn.Module):
    filters: int
    strides: tuple = (1, 1)
    use_projection: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        if self.use_projection:
            residual = nn.Conv(self.filters * 4, (1, 1), self.strides, use_bias=False)(
                x
            )
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        x = nn.Conv(self.filters, (1, 1), self.strides, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(self.filters, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(self.filters * 4, (1, 1), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        return nn.relu(x + residual)


class ResNet50(nn.Module):
    """
    Standard ResNet-50 implementation using Bottleneck Blocks.
    Uses the 1x1 -> 3x3 -> 1x1 bottleneck architecture to increase depth while
    managing computational cost. Follows the [3, 4, 6, 3] structure.

    Attributes:
        num_classes (int): Number of output classes. Default 1000.
    """

    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        def make_layer(filters, blocks, stride):
            layers = []
            layers.append(
                BottleneckBlock(filters, strides=(stride, stride), use_projection=True)
            )
            for _ in range(1, blocks):
                layers.append(BottleneckBlock(filters))
            return layers

        for block in make_layer(64, 3, stride=1):
            x = block(x, train)
        for block in make_layer(128, 4, stride=2):
            x = block(x, train)
        for block in make_layer(256, 6, stride=2):
            x = block(x, train)
        for block in make_layer(512, 3, stride=2):
            x = block(x, train)

        x = jnp.mean(x, axis=(1, 2))  # global average pooling
        x = nn.Dense(self.num_classes)(x)
        return x
