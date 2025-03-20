import functools
import logging

import jax
import jax.numpy as jnp
import optax
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState

from ae_utils.ae import AutoEncoder, mse_recon_loss
from ae_utils.losses import mse_loss_fn, triplet_loss_fn
from qdax.core.containers.l_value_archive import UnstructuredRepertoire
from qdax.custom_types import Params
from qdax.tasks.environments.bd_extractors import (
    AuroraExtraInfoNormalization,
    get_aurora_encoding,
)
from qdax.utils import train_seq2seq


def train_step_mse(
    state: TrainState,
    batch: jnp.ndarray,
    random_key: jax.random.PRNGKey,
    alpha: float,
):
    """
    Performs a training step using Mean Squared Error (MSE) loss.

    Args:
        state (TrainState): The current training state.
        batch (jnp.ndarray): Input batch of shape (batch_size, sequence_length, features).
        random_key (jax.random.PRNGKey): JAX random key.
        alpha (float): Hyperparameter (not used in MSE loss but kept for consistency).

    Returns:
        Tuple[TrainState, float, dict]: Updated training state, loss value, and loss dictionary.
    """
    lstm_key = jax.random.fold_in(random_key, state.step)
    dropout_key, lstm_key = jax.random.split(lstm_key, 2)

    # Shift input by one to avoid leakage
    batch_decoder = jnp.roll(batch, shift=1, axis=1)
    # Large number as zero token
    batch_decoder = batch_decoder.at[:, 0, :].set(-1000)

    def compute_loss(params: Params):
        logits, _ = state.apply_fn(
            {"params": params},
            batch,
            batch_decoder,
            rngs={"lstm": lstm_key, "dropout": dropout_key},
        )
        loss = mse_loss_fn(logits=logits, batch=batch_decoder)
        return loss, {"recon_loss": loss, "triplet_loss": 0.0, "model_loss": loss}

    (loss, loss_dict), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        state.params
    )
    state = state.apply_gradients(grads=grads)

    return state, loss, loss_dict


def train_step_triplet_mse_loss(
    state: TrainState,
    batch: jnp.ndarray,
    random_key: jax.random.PRNGKey,
    alpha: float,
    triplet_loss_weight: float,
):
    """
    Performs a training step using a combination of MSE loss and Triplet loss.

    Args:
        state (TrainState): The current training state.
        batch (jnp.ndarray): Input batch of shape (batch_size, 3, sequence_length, features) with (anchor, positive, negative).
        random_key (jax.random.PRNGKey): JAX random key.
        alpha (float): Triplet margin parameter.
        triplet_loss_weight (float): Weight for the triplet loss component.

    Returns:
        Tuple[TrainState, float, dict]: Updated training state, loss value, and loss dictionary.
    """
    lstm_key = jax.random.fold_in(random_key, state.step)
    dropout_key_1, dropout_key_2, dropout_key_3, lstm_key_1, lstm_key_2, lstm_key_3 = (
        jax.random.split(lstm_key, 6)
    )

    anchor_batch = batch[:, 0, :]
    pos_batch = batch[:, 1, :]
    neg_batch = batch[:, 2, :]

    # Shift input by one to avoid leakage
    anchor_decoder = jnp.roll(anchor_batch, shift=1, axis=1)
    pos_decoder = jnp.roll(pos_batch, shift=1, axis=1)
    neg_decoder = jnp.roll(neg_batch, shift=1, axis=1)
    # Large number as zero token
    anchor_decoder = anchor_decoder.at[:, 0, :].set(-1000)
    pos_decoder = pos_decoder.at[:, 0, :].set(-1000)
    neg_decoder = neg_decoder.at[:, 0, :].set(-1000)

    def compute_loss(params: Params):

        anchor_logits, anchor_latent = state.apply_fn(
            {"params": params},
            anchor_batch,
            anchor_decoder,
            rngs={"lstm": lstm_key_1, "dropout": dropout_key_1},
        )
        pos_logits, pos_latent = state.apply_fn(
            {"params": params},
            pos_batch,
            pos_decoder,
            rngs={"lstm": lstm_key_2, "dropout": dropout_key_2},
        )
        neg_logits, neg_latent = state.apply_fn(
            {"params": params},
            neg_batch,
            neg_decoder,
            rngs={"lstm": lstm_key_3, "dropout": dropout_key_3},
        )

        recon_loss = mse_loss_fn(logits=anchor_logits, batch=anchor_decoder)
        triplet_loss = jax.vmap(triplet_loss_fn, in_axes=(0, 0, 0, None))(
            anchor_latent, pos_latent, neg_latent, alpha
        )
        triplet_loss = jnp.sum(triplet_loss, axis=0)

        total_loss = triplet_loss_weight * triplet_loss + recon_loss
        return total_loss, {
            "recon_loss": recon_loss,
            "triplet_loss": triplet_loss,
            "model_loss": total_loss,
        }

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, loss_dict), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, loss_dict


def train_step_triplet_loss(
    state: TrainState,
    batch: jnp.ndarray,
    random_key: jax.random.PRNGKey,
    alpha: float,
):
    """
    Performs a training step using Triplet loss.

    Args:
        state (TrainState): The current training state.
        batch (jnp.ndarray): Input batch of shape (batch_size, 3, sequence_length, features).
        random_key (jax.random.PRNGKey): JAX random key.
        alpha (float): Triplet margin parameter.

    Returns:
        Tuple[TrainState, float, dict]: Updated training state, loss value, and loss dictionary.
    """
    lstm_key = jax.random.fold_in(random_key, state.step)
    dropout_key_1, dropout_key_2, dropout_key_3, lstm_key_1, lstm_key_2, lstm_key_3 = (
        jax.random.split(lstm_key, 6)
    )

    anchor_batch = batch[:, 0, :]
    pos_batch = batch[:, 1, :]
    neg_batch = batch[:, 2, :]

    # Shift input by one to avoid leakage
    anchor_decoder = jnp.roll(anchor_batch, shift=1, axis=1)
    pos_decoder = jnp.roll(pos_batch, shift=1, axis=1)
    neg_decoder = jnp.roll(neg_batch, shift=1, axis=1)
    # Large number as zero token
    anchor_decoder = anchor_decoder.at[:, 0, :].set(-1000)
    pos_decoder = pos_decoder.at[:, 0, :].set(-1000)
    neg_decoder = neg_decoder.at[:, 0, :].set(-1000)

    def compute_loss(params: Params):

        anchor_logits, anchor_latent = state.apply_fn(
            {"params": params},
            anchor_batch,
            anchor_decoder,
            rngs={"lstm": lstm_key_1, "dropout": dropout_key_1},
        )
        pos_logits, pos_latent = state.apply_fn(
            {"params": params},
            pos_batch,
            pos_decoder,
            rngs={"lstm": lstm_key_2, "dropout": dropout_key_2},
        )
        neg_logits, neg_latent = state.apply_fn(
            {"params": params},
            neg_batch,
            neg_decoder,
            rngs={"lstm": lstm_key_3, "dropout": dropout_key_3},
        )

        triplet_loss = jax.vmap(triplet_loss_fn, in_axes=(0, 0, 0, None))(
            anchor_latent, pos_latent, neg_latent, alpha
        )
        triplet_loss = jnp.sum(triplet_loss, axis=0)

        return triplet_loss, {
            "triplet_loss": triplet_loss,
            "recon_loss": 0.0,
            "model_loss": triplet_loss,
        }

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, loss_dict), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, loss_dict


def train_step_ae(
    state: TrainState,
    batch: jnp.ndarray,
    random_key: jax.random.PRNGKey,
    alpha: float,
    model,
):
    
    # Get loss and gradients for loss
    loss, grads = jax.value_and_grad(mse_recon_loss)(model, state.params, batch) 
    loss_dict = {"recon_loss": loss, "triplet_loss": 0.0, "model_loss": loss}

    state = state.apply_gradients(grads=grads)  # Optimizer update step
    return state, loss, loss_dict


def get_triplet_batch(repertoire, batch_indices, random_key):
    """
    Generates triplets (anchor, positive, negative) based on fitness distances.

    Args:
        repertoire: The dataset containing fitness values.
        batch_indices (jnp.ndarray): Indices of batch samples.
        random_key (jax.random.PRNGKey): Random key for sampling.

    Returns:
        jnp.ndarray: Array of shape (batch_size, 3) containing anchor, positive, and negative indices.
    """
    # Compute the distance matrix over batch individuals
    batch_fitnesses = repertoire.fitnesses[batch_indices]
    distances = jnp.abs(batch_fitnesses[:, None] - batch_fitnesses[None, :])

    # Mask diagonal elements (self-distance) to prevent self-selection
    mask = jnp.where(distances == 0, 0, 1)

    keys = jax.random.split(random_key, batch_indices.shape[0])

    def select_random_pairs(idx, rng_key):
        """
        Selects two distinct indices for each sample in the batch.

        Args:
            idx (int): Index of the sample.
            rng_key (jax.random.PRNGKey): Random key for JAX operations.

        Returns:
            jnp.ndarray: Two randomly chosen indices (potential positive and negative).
        """
        return jax.random.choice(
            a=distances[idx].shape[0],
            shape=(2,),
            replace=False,
            p=mask[idx],
            key=rng_key,
        )
    
    random_pair_indices = jax.vmap(select_random_pairs)(jnp.arange(batch_indices.shape[0]), keys)

    # Gather the distances for the chosen pairs
    chosen_distances = distances[
        jnp.arange(batch_indices.shape[0])[:, None], random_pair_indices
    ]

    # Identify positive (smallest distance) and negative (largest distance)
    pos_indices = jnp.argmin(chosen_distances, axis=-1)
    neg_indices = jnp.argmax(chosen_distances, axis=-1)

    # Generate triplet indices
    positive_indices = batch_indices[
        random_pair_indices[jnp.arange(batch_indices.shape[0]), pos_indices]
    ]
    negative_indices = batch_indices[
        random_pair_indices[jnp.arange(batch_indices.shape[0]), neg_indices]
    ]
    triplets = jnp.stack([batch_indices, positive_indices, negative_indices], axis=1)
    return triplets

def normalize_observations(repertoire: UnstructuredRepertoire, is_image_obs: bool):
    """
    Computes mean and standard deviation of the observations for normalization.
    
    Args:
        repertoire (UnstructuredRepertoire): The dataset containing observations.
        is_image_obs (bool): Whether the observations are images.
        
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Normalised dataset, mean and 
        standard deviation of the observations."""

    obs_mean = jnp.nanmean(repertoire.observations, axis=(0, 1) if not is_image_obs else 0)
    obs_std = jnp.nanstd(repertoire.observations, axis=(0, 1) if not is_image_obs else 0)
    obs_std = jnp.where(obs_std == 0, 1, obs_std)  # Avoid division by zero
    normalized_dataset = (repertoire.observations - obs_mean) / obs_std
    return normalized_dataset, obs_mean, obs_std

def train_function(
    repertoire: UnstructuredRepertoire,
    train_state: TrainState,
    random_key: jax.random.PRNGKey,
    model,
    model_epoch_period: int,
    learning_rate: float,
    batch_size: int,
    is_image_obs: bool,
    train_step_fn,
    reinit_opt: bool,
    min_delta: float = 10,
    use_triplet_loss: bool = False,
    margin_auto_adjust: bool = False,
    hidden_size: int = 10,
    alpha: float = 0.2,
) -> AuroraExtraInfoNormalization:
    
    """
    Trains a model using either MSE loss or Triplet loss with optional early stopping.

    Args:
        repertoire (UnstructuredRepertoire): Data container for training.
        train_state (TrainState): Current training state.
        random_key (jax.random.PRNGKey): Random key for JAX operations.
        model: The model being trained.
        model_epoch_period (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Size of each training batch.
        is_image_obs (bool): If True, observations are images, otherwise sequences.
        train_step_fn: Function defining the training step.
        reinit_opt (bool): If True, reinitializes optimizer before training.
        min_delta (float, optional): Minimum delta for early stopping. Defaults to 10.
        use_triplet_loss (bool, optional): If True, enables triplet loss. Defaults to False.
        margin_auto_adjust (bool, optional): If True, adjusts margin dynamically. Defaults to False.
        hidden_size (int, optional): Hidden size used for dynamic margin adjustment. Defaults to 10.
        alpha (float, optional): Margin parameter for triplet loss. Defaults to 0.2.

    Returns:
        Tuple[TrainState, AuroraExtraInfoNormalization, dict]: Updated model state, normalization info, and training metrics.
    """

    # Normalize observations
    dataset, obs_mean, obs_std = normalize_observations(repertoire, is_image_obs)

    # Option to reinitialize optimizer
    state = (
        TrainState.create(apply_fn=model.apply, params=train_state.params, tx=optax.adam(learning_rate))
        if reinit_opt
        else train_state
    )

    # Setup early stopping
    early_stop = EarlyStopping(min_delta=min_delta, patience=10)

    steps_per_epoch = repertoire.max_size // batch_size

    # Shuffling indices of valid individuals in the repertoire
    random_key, subkey = jax.random.split(random_key)
    is_valid = repertoire.fitnesses != -jnp.inf
    indices = jax.random.choice(
        subkey,
        jnp.arange(repertoire.max_size),
        shape=(repertoire.max_size,),
        p=is_valid,
    )

    # Triplet loss generates triplets for each batch
    if use_triplet_loss:
        random_key, subkey = jax.random.split(random_key)
        indices = get_triplet_batch(repertoire, indices, subkey)

    losses, triplet_losses, recon_losses = [], [], []

    for epoch in range(model_epoch_period):

        # Shuffle indices for this epoch
        random_key, subkey = jax.random.split(random_key)
        epoch_indices = jax.random.permutation(subkey, indices)
        epoch_indices = epoch_indices.at[: steps_per_epoch * batch_size].get()

        if use_triplet_loss:
            epoch_indices = jnp.reshape(epoch_indices, (steps_per_epoch, batch_size, 3))
        else:
            epoch_indices = jnp.reshape(epoch_indices, (steps_per_epoch, batch_size))

        total_loss = 0.0
        triplet_loss = 0.0
        recon_loss = 0.0

        for b_idx in epoch_indices:
            batch = dataset.at[b_idx].get()
            random_key, subkey = jax.random.split(random_key)

            # Adjust alpha (triplet margin) dynamically
            if margin_auto_adjust:
                if hasattr(repertoire, "d_min"):
                    alpha = hidden_size * repertoire.d_min.copy()
                elif hasattr(repertoire, "l_value"):
                    alpha = hidden_size * repertoire.l_value.copy()
            
            state, loss_val, loss_dict = train_step_fn(
                    state, batch, subkey, alpha=alpha
                )
            
            total_loss += loss_val
            triplet_loss += loss_dict["triplet_loss"]
            recon_loss += loss_dict["recon_loss"]

        # Compute average loss for the epoch
        avg_loss = total_loss / steps_per_epoch
        logging.info("Epoch: {}, loss: {:.4f}".format(epoch + 1, avg_loss))
        
        losses.append(avg_loss)
        triplet_losses.append(triplet_loss / steps_per_epoch)
        recon_losses.append(recon_loss / steps_per_epoch)

        # Early stopping condition
        early_stop = early_stop.update(avg_loss)
        if early_stop.should_stop:
            logging.info("Early stopping at epoch: {}".format(epoch))
            break

    # Store training metrics
    metrics = {
        "model_loss": jnp.array(losses),
        "recon_loss": jnp.array(recon_losses),
        "triplet_loss": jnp.array(triplet_losses),
    }

    aurora_extra_info = AuroraExtraInfoNormalization.create(
        model_params=state.params,
        mean_observations=obs_mean,
        std_observations=obs_std,
    )

    return state, aurora_extra_info, metrics


def init_autoencoder_model_training(cfg, observations_dims, random_key):
    """
    Initializes the autoencoder model training setup, including model selection, 
    loss function assignment, optimizer initialization, and function bindings.

    Args:
        cfg: Configuration object containing model and training parameters.
        observations_dims (Tuple[int]): The shape of the observation data.
        random_key (jax.random.PRNGKey): Random key for model initialization.

    Returns:
        Tuple:
            - encoder_fn (Callable): JIT-compiled function to encode input observations.
            - train_fn (Callable): Function to execute model training.
            - train_state (TrainState): Initial training state with model parameters and optimizer.
            - aurora_extra_info (AuroraExtraInfoNormalization): Object containing normalization details.

    The function performs the following:
        1. Splits the JAX random key to initialize the model.
        2. Determines the model type:
            - If observations are images, initializes an `AutoEncoder`.
            - Otherwise, initializes a sequence-to-sequence model (`train_seq2seq`).
        3. Initializes model parameters and constructs the loss function.
        4. Creates an optimizer (Adam) and sets up the training state.
        5. Defines:
            - `encoder_fn`: A function to encode observations using the trained model.
            - `train_fn`: A function to execute training over multiple epochs.
        6. Returns the encoder function, training function, training state, and normalization details.
    """

    random_key, subkey = jax.random.split(random_key)

    if cfg.env.observation_extraction.observation_option == "images":
        random_key, subkey = jax.random.split(random_key)
        model = AutoEncoder(
            img_shape=observations_dims,
            latent_size=cfg.hidden_size,
            features=cfg.features,
        )
        model_params = model.init(subkey, jnp.ones((1, *observations_dims)))["params"]
        aurora_extra_info = AuroraExtraInfoNormalization.create(
            model_params,
            jnp.zeros(observations_dims),
            jnp.ones(observations_dims),
        )
        train_step_fn = jax.jit(functools.partial(
            train_step_ae,
            model=model,
        ))

    else:
        model = train_seq2seq.get_model(
            observations_dims[-1], True, hidden_size=cfg.hidden_size
        )
        model_params = train_seq2seq.get_initial_params(
            model, subkey, (1, *observations_dims)
        )
        aurora_extra_info = AuroraExtraInfoNormalization.create(
            model_params,
            jnp.zeros(observations_dims[-1]),
            jnp.ones(observations_dims[-1]),
        )

        if cfg.loss_type == "mse":
            train_step_fn = jax.jit(train_step_mse)
        elif cfg.loss_type == "triplet":
            train_step_fn = jax.jit(train_step_triplet_loss)
        elif cfg.loss_type == "both":
            train_step_fn = jax.jit(
                functools.partial(train_step_triplet_mse_loss, triplet_loss_weight=cfg.triplet_loss_weight)
            )


    optimizer = optax.adam(cfg.learning_rate)
    train_state = TrainState.create(
        apply_fn=model.apply, params=model_params, tx=optimizer
    )

    # Define the encoder function
    encoder_fn = jax.jit(
        functools.partial(
            get_aurora_encoding,
            model=model,
        )
    )

    # Define the training function
    train_fn = functools.partial(
        train_function,
        model=model,
        model_epoch_period=cfg.model_epoch_period,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.model_batch_size,
        is_image_obs=cfg.env.observation_extraction.observation_option == "images",
        train_step_fn=train_step_fn,
        reinit_opt=cfg.reinit_opt,
        min_delta=cfg.min_delta_early_stopping,
        use_triplet_loss=cfg.loss_type in ["triplet", "both"],
        margin_auto_adjust=cfg.margin_auto_adjust,
        hidden_size=cfg.hidden_size,
        alpha=cfg.triplet_margin,
    )

    return encoder_fn, train_fn, train_state, aurora_extra_info
