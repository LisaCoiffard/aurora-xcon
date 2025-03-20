from typing import Tuple

import flax.linen as nn


class Encoder(nn.Module):
    latent_size: int
    features: int
    initializer: nn.initializers.Initializer = nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, x):
        # First convolution block (28x28x1 -> 14x14xfeatures)
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=self.initializer,
        )(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Second convolution block (14x14xfeatures -> 7x7x(2*features))
        x = nn.Conv(
            features=2 * self.features,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=self.initializer,
        )(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten and project to latent space
        x = x.reshape((x.shape[0], -1))  # Flatten to (batch_size, 7*7*2*features)
        x = nn.Dense(features=self.latent_size, kernel_init=self.initializer)(x)
        return x


class Decoder(nn.Module):
    img_shape: Tuple[int, int, int]
    features: int
    initializer: nn.initializers.Initializer = nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, x):
        # Project and reshape
        initial_features = 2 * self.features
        x = nn.Dense(features=7 * 7 * initial_features, kernel_init=self.initializer)(x)
        x = nn.gelu(x)
        x = x.reshape(
            (-1, 7, 7, initial_features)
        )  # Reshape to (batch, 7, 7, 2*features)

        # First upsample block (7x7x(2*features) -> 14x14xfeatures)
        x = nn.ConvTranspose(
            features=self.features,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            kernel_init=self.initializer,
        )(x)
        x = nn.gelu(x)

        # Second upsample block (14x14xfeatures -> 28x28x1)
        x = nn.ConvTranspose(
            features=self.img_shape[
                -1
            ],  # Output channels should match input (1 for MNIST)
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            kernel_init=self.initializer,
        )(x)
        return x


class AutoEncoder(nn.Module):
    img_shape: Tuple[int, int, int]
    latent_size: int
    features: int
    initializer: nn.initializers.Initializer = nn.initializers.lecun_uniform()

    def setup(self):
        self.encoder = Encoder(
            latent_size=self.latent_size,
            features=self.features,
            initializer=self.initializer,
        )
        self.decoder = Decoder(
            img_shape=self.img_shape,
            features=self.features,
            initializer=self.initializer,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def mse_recon_loss(model, params, batch):
    recon_imgs = model.apply({"params": params}, batch)
    loss = (
        ((recon_imgs - batch) ** 2).mean(axis=0).sum()
    )  # Mean over batch, sum over pixels
    return loss
