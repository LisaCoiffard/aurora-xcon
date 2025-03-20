import functools
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from omegaconf import DictConfig
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.mdp_utils import generate_unroll
from qdax.custom_types import (
    Descriptor,
    EnvState,
    ExtraScores,
    Fitness,
    Genotype,
    Params,
    RNGKey,
)
from qdax.tasks.brax_envs import get_mask_from_transitions


def compute_fitnesses(
    data: QDTransition, mask: jnp.array, reward_type: str
) -> Tuple[jnp.array, jnp.array]:
    """
    Compute fitnesses for the active and passive archive.

    The fitnesses are computed as the sum of the rewards over the episode. If
    buffer is used, the rewards are summed over episode after the buffer has ended.

    Args:
        data: QDTransition namedtuple containing the data from the rollouts.
        mask: Mask over buffer time (if applicable) and non-terminated episode length.
        done_mask: Mask over non-terminated episode length.
        reward_type: Type of reward to use for computing the fitnesses. Can be "full"
            (use normal ant environment reward), "no_forward" (ant env reward without
            forward term) or "forward_only" (only forward term of ant env reward).

    Returns:
        active_fitnesses: Fitnesses for the active archive.
        forward_reward: Fitnesses for the passive archive.
    """
    # get fitness scores from reward
    if reward_type == "full":
        fitnesses = jnp.sum(data.rewards[:, :, 0] * (1.0 - mask), axis=1)
    elif reward_type == "no_forward":
        fitnesses = jnp.sum(data.rewards[:, :, 1] * (1.0 - mask), axis=1)
    elif reward_type == "forward_only":
        fitnesses = jnp.sum(data.rewards[:, :, 2] * (1.0 - mask), axis=1)
    elif reward_type == "final":
        last_index = jnp.int32(jnp.sum(1.0 - mask, axis=1)) - 1
        fitnesses = jax.vmap(lambda x, y: x[y])(data.rewards, last_index)
    else:
        fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)

    return fitnesses


@partial(
    jax.jit,
    static_argnames=(
        "cfg",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def scoring_function(
    policies_params: Genotype,
    random_key: RNGKey,
    init_states: EnvState,
    cfg: DictConfig,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:

    # Perform rollouts with each policy
    random_key, subkey = jax.random.split(random_key)
    unroll_fn = partial(
        generate_unroll,
        episode_length=cfg.env.episode_length,
        play_step_fn=play_step_fn,
        random_key=subkey,
    )

    _final_state, data = jax.vmap(unroll_fn)(init_states, policies_params)

    # create a mask to extract data properly
    mask = get_mask_from_transitions(data)

    # extract behaviour descriptors
    descriptors = behavior_descriptor_extractor(data, mask)

    # compute fitnesses
    fitnesses = compute_fitnesses(
        data,
        mask,
        cfg.env.reward_type,
    )

    return (
        fitnesses,
        descriptors,
        {
            "transitions": data,
            "final_state": _final_state,
            "mask": mask,
        },
        random_key,
    )


def passive_scoring_function(
    scoring_fn: Callable,
    passive_behaviour_descriptor_extractor: Callable[
        [QDTransition, jnp.ndarray], Descriptor
    ],
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:

    @functools.wraps(scoring_fn)
    def _wrapper(
        policies_params: Genotype,
        random_key: RNGKey,
    ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:

        # Call scoring_function
        active_fitnesses, active_descriptors, extra_scores, random_key = scoring_fn(
            policies_params, random_key
        )

        data = extra_scores["transitions"]
        final_state = extra_scores["final_state"]
        mask = extra_scores["mask"]

        # extract behaviour descriptors
        passive_descriptors = passive_behaviour_descriptor_extractor(data, mask)

        # passive fitnesses are the same as active fitnesses
        passive_fitnesses = active_fitnesses

        return (
            active_fitnesses,
            active_descriptors,
            {
                "transitions": data,
                "final_state": final_state,
                "passive_descriptors": passive_descriptors,
                "passive_fitnesses": passive_fitnesses,
            },
            random_key,
        )

    return _wrapper


@partial(
    jax.jit,
    static_argnames=(
        "cfg",
        "play_step_fn",
        "play_reset_fn",
        "behavior_descriptor_extractor",
        "passive_behavior_descriptor_extractor",
    ),
)
def reset_based_scoring_function(
    policies_params: Genotype,
    random_key: RNGKey,
    cfg: DictConfig,
    play_reset_fn: Callable[[RNGKey], EnvState],
    play_step_fn: Callable[
        [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
    passive_behavior_descriptor_extractor: Callable[
        [QDTransition, jnp.ndarray], Descriptor
    ],
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(
        subkey, jax.tree_util.tree_leaves(policies_params)[0].shape[0]
    )
    reset_fn = jax.vmap(play_reset_fn)
    init_states = reset_fn(keys)

    scoring_fn = partial(
        scoring_function,
        init_states=init_states,
        cfg=cfg,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=behavior_descriptor_extractor,
    )

    if passive_behavior_descriptor_extractor:
        wrapped_scoring_fn = passive_scoring_function(
            scoring_fn,
            passive_behavior_descriptor_extractor,
        )

    else:
        wrapped_scoring_fn = scoring_fn

    fitnesses, descriptors, extra_scores, random_key = wrapped_scoring_fn(
        policies_params, random_key
    )

    return fitnesses, descriptors, extra_scores, random_key
