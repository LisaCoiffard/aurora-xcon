import jax
import jax.numpy as jnp
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Descriptor


def get_bumper_contacts(
    data: QDTransition,
    mask: jnp.ndarray,
) -> Descriptor:
    """Compute feet contact time proportion."""
    data = data.state_desc
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    bumper_contacts = data[:, :, 2:4]

    bumper_contacts = jnp.where(bumper_contacts < 0, 0, 1)

    descriptors = jnp.nansum(bumper_contacts * (1.0 - mask), axis=1)
    descriptors = descriptors / jnp.nansum(1.0 - mask, axis=1)

    return descriptors


def get_mean_laser_measures(
    data: QDTransition,
    mask: jnp.ndarray,
) -> Descriptor:
    """Compute feet contact time proportion."""
    data = data.state_desc
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    data = data[:, :, 4:]

    descriptors = jnp.nansum(data * (1.0 - mask), axis=1)
    descriptors = descriptors / jnp.nansum(1.0 - mask, axis=1)

    return descriptors


def get_random_descriptor_old(
    data: QDTransition,
    mask: jnp.ndarray,
    bd_dims: tuple,
    is_binary: jnp.ndarray,
    min_bd: jnp.ndarray,
    max_bd: jnp.ndarray,
) -> Descriptor:
    """Compute random descriptor.

    Args:
        data: QDTransition containing observations of shape (batch_size, traj_length, obs_dim)
        mask: Binary mask of shape (batch_size, traj_length), where 1 indicates trajectory has ended
        bd_dims: Tuple of 2 integers indicating which observation dimensions to use as descriptors
        is_binary: Array of 2 booleans indicating if each descriptor dimension is binary
        min_bd: Minimum values for each descriptor dimension, shape (2,)
        max_bd: Maximum values for each descriptor dimension, shape (2,)
    Returns:
        Array of shape (batch_size, 2) containing computed descriptors. For binary dimensions,
        returns proportion of 1s over unmasked trajectory. For non-binary dimensions, returns
        the last unmasked value.
    """
    data = data.obs
    mask = jnp.expand_dims(mask, axis=-1)

    selected_descriptors = data[:, :, bd_dims]

    # Compute proportions for binary dimensions
    binary_proportions = jnp.nansum(selected_descriptors * (1.0 - mask), axis=1) / jnp.nansum(
        1.0 - mask, axis=1
    )

    # Get last unmasked value for non-binary dimensions
    last_index = jnp.int32(jnp.nansum(1.0 - mask, axis=1)) - 1
    last_values = jax.vmap(lambda x, y: x[y])(selected_descriptors, last_index).squeeze(
        axis=1
    )

    # Combine results based on binary detection
    descriptors = jnp.where(
        is_binary,
        binary_proportions,
        last_values,
    )
    descriptors = jnp.clip(descriptors, min_bd, max_bd)

    return descriptors


def get_random_descriptor(
    data: QDTransition,
    mask: jnp.ndarray,
    bd_dims: tuple,
    min_bd: jnp.ndarray,
    max_bd: jnp.ndarray,
    random_key: jnp.ndarray,
) -> Descriptor:
    """Compute random descriptor.

    Args:
        data: QDTransition containing observations of shape (batch_size, traj_length, obs_dim)
        mask: Binary mask of shape (batch_size, traj_length), where 1 indicates trajectory has ended
        bd_dims: Tuple of 2 integers indicating which observation dimensions to use as descriptors
        min_bd: Minimum values for each descriptor dimension, shape (2,)
        max_bd: Maximum values for each descriptor dimension, shape (2,)
    Returns:
        Array of shape (batch_size, 2) containing computed descriptors. For binary dimensions,
        returns proportion of 1s over unmasked trajectory. For non-binary dimensions, returns
        the last unmasked value.
    """
    random_key, _ = jax.random.split(random_key)
    data = data.obs
    selected_descriptors = data[:, :, bd_dims]

    traj_lengths = jnp.sum(1.0 - mask, axis=1)
    random_step_idx = jax.random.randint(random_key, (data.shape[0],), 0, traj_lengths)
    descriptors = jax.vmap(lambda x, y: x[y])(selected_descriptors, random_step_idx)

    descriptors = jnp.clip(descriptors, min_bd, max_bd)

    return descriptors


def get_feet_contact_proportion(
    data: QDTransition, mask: jnp.ndarray, feet_name: str
) -> Descriptor:
    """Compute feet contact time proportion."""
    data = data.state_desc
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    if feet_name == "front":
        feet_contact = data[:, :, :2]
    elif feet_name == "back":
        feet_contact = data[:, :, 2:4]
    elif feet_name == "all":
        feet_contact = data[:, :, :4]
    else:
        raise ValueError(f"Feet name {feet_name} not recognized.")

    descriptors = jnp.nansum(feet_contact * (1.0 - mask), axis=1)
    descriptors = descriptors / jnp.nansum(1.0 - mask, axis=1)

    return descriptors


def get_final_xy_position(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Compute final xy positon.

    This function suppose that state descriptor is the xy position, as it
    just select the final one of the state descriptors given.
    """
    data = data.state_desc
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    data = data[:, :, :2]

    # Get behavior descriptor
    last_index = jnp.int32(jnp.nansum(1.0 - mask, axis=1)) - 1
    descriptors = jax.vmap(lambda x, y: x[y])(data, last_index)

    # remove the dim coming from the trajectory
    return descriptors.squeeze(axis=1)
