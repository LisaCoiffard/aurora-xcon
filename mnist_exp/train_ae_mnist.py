import logging
import os

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow_datasets as tfds
import wandb
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA

from ae_utils.ae import AutoEncoder
from ae_utils.losses import triplet_loss_fn


def get_datasets():
    """Load MNIST dataset and return dataset info."""
    data_dir = os.path.join(get_original_cwd(), "mnist_exp/data")
    logging.info(f"Data directory: {data_dir}")
    mnist_data, info = tfds.load(
        "mnist", batch_size=-1, data_dir=data_dir, with_info=True, shuffle_files=True
    )
    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data["train"], mnist_data["test"]
    num_labels = info.features["label"].num_classes
    h, w, c = info.features["image"].shape

    info = {
        "num_labels": num_labels,
        "img_shape": (h, w, c),
    }

    train_images, train_labels = train_data["image"], train_data["label"]
    test_images, test_labels = test_data["image"], test_data["label"]

    def normalize_img(image):
        return jnp.float32(image) / 255.0

    train_images = normalize_img(train_images)
    test_images = normalize_img(test_images)

    return train_images, train_labels, test_images, test_labels, info


@jax.jit
def train_step_mse(state: TrainState, batch_imgs):
    """Training step for MSE loss."""

    def compute_loss(params):
        recon_imgs, _ = state.apply_fn({"params": params}, batch_imgs)
        loss = ((recon_imgs - batch_imgs) ** 2).mean()
        return loss, {"recon_loss": loss, "model_loss": loss}

    (loss, metrics), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        state.params
    )
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics


@jax.jit
def train_step_triplet(state: TrainState, batch: jnp.ndarray, alpha: float):
    """Training step for triplet loss."""
    anchor = batch[:, 0]
    positive = batch[:, 1]
    negative = batch[:, 2]

    def compute_loss(params):
        _, anchor_latent = state.apply_fn(
            {"params": params},
            anchor,
        )
        _, pos_latent = state.apply_fn(
            {"params": params},
            positive,
        )
        _, neg_latent = state.apply_fn(
            {"params": params},
            negative,
        )

        triplet_loss = jax.vmap(triplet_loss_fn, in_axes=(0, 0, 0, None))(
            anchor_latent, pos_latent, neg_latent, alpha
        )
        triplet_loss = jnp.mean(triplet_loss)

        # Also compute reconstruction loss for monitoring
        recon_anchor, _ = state.apply_fn({"params": params}, anchor)
        recon_loss = ((recon_anchor - anchor) ** 2).mean()

        return triplet_loss, {
            "triplet_loss": triplet_loss,
            "recon_loss": recon_loss,
            "model_loss": triplet_loss,
        }

    (loss, metrics), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        state.params
    )
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics


def get_batches(images, labels, batch_size, key):
    """Create shuffled batches from images and labels."""
    num_samples = len(images)
    num_batches = num_samples // batch_size

    # Shuffle indices
    shuffled_idx = jax.random.permutation(key, jnp.arange(num_samples))

    # Truncate to multiple of batch_size
    truncated_idx = shuffled_idx[: num_batches * batch_size]

    # Reshape into batches
    batch_idx = truncated_idx.reshape((num_batches, batch_size))

    # Create batches using fancy indexing
    image_batches = images[batch_idx]
    label_batches = labels[batch_idx]

    return image_batches, label_batches, num_batches


def get_triplet_batches(images, labels, batch_size, key):
    """Create shuffled triplet batches from images and labels."""
    # First get regular batches
    image_batches, label_batches, num_batches = get_batches(
        images, labels, batch_size, key
    )

    triplet_batches = []
    # Create triplets for each batch
    for batch_idx in range(num_batches):
        batch_images = image_batches[batch_idx]
        batch_labels = label_batches[batch_idx]

        key, subkey = jax.random.split(key)
        triplet_batch = get_triplets(batch_images, batch_labels, subkey)
        triplet_batches.append(triplet_batch)

    return jnp.stack(triplet_batches), num_batches


def get_triplets(images, labels, key):
    """Vectorized implementation of triplet generation."""
    n_samples = len(labels)

    # Create matrix of all pairs of label comparisons
    label_matches = labels[:, None] == labels[None, :]
    label_mismatches = ~label_matches

    # Mask out self-comparisons for positive pairs
    positive_mask = label_matches & ~jnp.eye(n_samples, dtype=bool)

    random_key, key1, key2 = jax.random.split(key, 3)

    def choose_masked_indices(mask, key):
        # Create random probabilities for each position
        probs = jax.random.uniform(key, (n_samples, n_samples))
        # Set probability to -1 for invalid positions (will never be selected)
        masked_probs = jnp.where(mask, probs, -1.0)
        # Take argmax along axis 1 to select one index per row
        return jnp.argmax(masked_probs, axis=1)

    # Choose positive and negative samples
    pos_indices = choose_masked_indices(positive_mask, key1)
    neg_indices = choose_masked_indices(label_mismatches, key2)

    # Gather the images
    anchor_images = images
    positive_images = images[pos_indices]
    negative_images = images[neg_indices]

    # Stack the triplets
    triplets = jnp.stack([anchor_images, positive_images, negative_images], axis=1)

    return triplets


def plot_reconstructions(model, params, images, labels=None):
    """Plot one example reconstruction per digit class."""
    if labels is None:
        # If no labels provided, use first 10 images
        samples = images[:10]
    else:
        # Get one example per class
        examples = {i: None for i in range(10)}
        for img, label in zip(images, labels):
            if examples[label] is None:
                examples[label] = img
            if all(v is not None for v in examples.values()):
                break
        samples = np.stack(list(examples.values()))

    # Create reconstructions
    recons, _ = model.apply({"params": params}, samples)

    # Plot
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        # Original
        axes[0, i].imshow(samples[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")

        # Reconstruction
        axes[1, i].imshow(recons[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")

    return fig


def plot_latent_space(model, params, images, labels, key):
    """Plot latent space embeddings colored by digit class."""
    # Get embeddings
    embeddings = model.apply({"params": params}, images, method=model.encode)

    # Use PCA if latent dim > 2
    if embeddings.shape[1] > 2:
        embeddings = PCA(n_components=2).fit_transform(embeddings)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()  # Get current axes
    
    if labels is not None:
        # Define colors and create scatter plot
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        scatter_points = []
        legend_labels = []

        # Plot each digit class separately
        for i in range(10):
            mask = labels == i
            if np.any(mask):
                scatter = plt.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=[colors[i]],
                    label=f"{i}",
                    s=20
                )
                scatter_points.append(scatter)
                legend_labels.append(f"{i}")

        # Remove box and axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # Add legend with larger markers and place it outside
        plt.legend(
            fontsize=25,
            markerscale=4,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            frameon=False  # Remove legend box
        )

        plt.tight_layout()

    return fig


def plot_latent_space_digits(model, params, images, labels, key):
    """Plot latent space embeddings showing digit images colored by class."""
    # Get embeddings
    embeddings = model.apply({"params": params}, images, method=model.encode)

    # Use PCA if latent dim > 2
    if embeddings.shape[1] > 2:
        embeddings = PCA(n_components=2).fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set plot limits with some margins
    x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
    margin = 0.1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Calculate image size based on data range
    x_range = x_max - x_min
    y_range = y_max - y_min
    img_size = min(x_range, y_range) * 0.04

    # Colors for different classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i in range(10):
        mask = labels == i
        if np.any(mask):
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[colors[i]],
                alpha=0.3,
                s=10
            )

    # Plot each digit as a small image
    for img, embedding, label in zip(images, embeddings, labels):
        # Calculate position in data coordinates
        x, y = embedding[0], embedding[1]
        
        # Create inset axes using data coordinates
        ax_inset = ax.inset_axes(
            [x-img_size/2, y-img_size/2, img_size, img_size],
            transform=ax.transData
        )

        # Reshape image and create RGBA array
        digit_img = img.reshape(28, 28)
        
        # Create RGBA array with transparency
        rgba_img = np.zeros((28, 28, 4))
        # Set RGB channels to the class color where digit is present
        for c in range(3):
            rgba_img[:, :, c] = colors[label][c] * (digit_img > 0.5)
        # Set alpha channel - make digit opaque and background transparent
        rgba_img[:, :, 3] = (digit_img > 0.5).astype(float)

        # Plot the digit image
        ax_inset.imshow(rgba_img, interpolation='bilinear')
        
        # Remove ticks and spines from inset axes
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        for spine in ax_inset.spines.values():
            spine.set_visible(False)

    # Set main axes properties and limits
    plt.title("Latent Space Visualization")
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    ax.set_xticks([])
    ax.set_yticks([])

    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors[i], label=f"Digit {i}", linewidth=2)
        for i in range(10)
    ]
    plt.legend(
        handles=legend_elements,
        title="Digit Classes",
    )

    return fig


@hydra.main(config_path="configs", config_name="mnist", version_base=None)
def train(cfg: DictConfig) -> None:

    logging.info(cfg)
    logging.info("Training")
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(config=wandb_cfg, **cfg.wandb):
        # Load dataset
        train_images, train_labels, test_images, test_labels, info = get_datasets()
        # Use only a subset of training data
        random_key = jax.random.PRNGKey(cfg.seed)
        # Shuffle and select subset
        random_key, subkey = jax.random.split(random_key)
        perm = jax.random.permutation(subkey, len(train_images))
        train_images, train_labels = train_images[perm], train_labels[perm]
        train_images, train_labels = train_images[:10000], train_labels[:10000]
        logging.info(
            f"Train images: {train_images.shape}, Test images: {test_images.shape}"
        )
        logging.info(f"Image shape: {info['img_shape']}")

        # Initialize model
        model = AutoEncoder(
            img_shape=info["img_shape"],
            latent_size=cfg.hidden_size,
            features=cfg.features,
        )

        # Initialize training state
        random_key, subkey = jax.random.split(random_key)
        dummy_input = jnp.ones((1, *info["img_shape"]))
        params = model.init(subkey, dummy_input)["params"]

        optimizer = optax.adam(cfg.learning_rate)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

        # Training loop
        early_stop = EarlyStopping(min_delta=cfg.min_delta_early_stopping, patience=10)

        for epoch in range(cfg.model_epoch_period):
            epoch_loss = 0
            batch_metrics = []

            random_key, subkey = jax.random.split(random_key)
            if cfg.loss_type == "triplet":
                batches, num_batches = get_triplet_batches(
                    train_images, train_labels, cfg.model_batch_size, subkey
                )

            else:
                batches, _, num_batches = get_batches(
                    train_images, train_labels, cfg.model_batch_size, subkey
                )

            for batch_idx in range(num_batches):
                batch = batches[batch_idx]
                if cfg.loss_type == "triplet":
                    state, loss, metrics = train_step_triplet(
                        state, batch, cfg.triplet_margin
                    )
                else:  # mse loss
                    state, loss, metrics = train_step_mse(state, batch)

                epoch_loss += loss
                batch_metrics.append(metrics)

            avg_loss = epoch_loss / num_batches
            avg_metrics = jax.tree.map(
                lambda *xs: jnp.mean(jnp.stack(xs)), *batch_metrics
            )
            logging.info(f"Epoch {epoch} - Loss: {avg_loss}")
            wandb.log(
                {
                    "epoch": epoch,
                    **{k: v.mean() for k, v in avg_metrics.items()},
                }
            )

            early_stop = early_stop.update(avg_loss)
            if early_stop.should_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

        # Log reconstructions
        recon_fig = plot_reconstructions(model, state.params, test_images)
        wandb.log({"reconstructions": wandb.Image(recon_fig)})
        plt.close(recon_fig)

        # Log latent space visualization
        random_key, subkey = jax.random.split(random_key)
        perm = jax.random.permutation(subkey, len(test_images))
        test_images, test_labels = test_images[perm], test_labels[perm]
        test_images, test_labels = test_images[:2000], test_labels[:2000]
        
        latent_fig = plot_latent_space_digits(
            model, state.params, test_images, test_labels, subkey
        )
        wandb.log({"latent_space_digits": wandb.Image(latent_fig)})
        plt.savefig("latent_space_digits.png")
        plt.close(latent_fig)

        latent_fig = plot_latent_space(
            model, state.params, test_images, test_labels, subkey
        )
        wandb.log({"latent_space": wandb.Image(latent_fig)})
        plt.savefig("latent_space.png")
        plt.close(latent_fig)


if __name__ == "__main__":
    train()
