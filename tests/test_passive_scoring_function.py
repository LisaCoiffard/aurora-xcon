from math import e, exp
import jax.numpy as jnp
import jax
import numpy as np
from functools import wraps
from collections import namedtuple
from dataclasses import dataclass
from tasks.scoring import compute_fitnesses

# Define test data and functions
QDTransition = namedtuple("QDTransition", ["rewards", "passive_desc", "state_desc"])
Genotype = object
RNGKey = object
Descriptor = object
Fitness = object
ExtraScores = object


# Define a dummy scoring function for testing
def scoring_fn_rdm(policies_params, random_key):
    active_fitnesses = jnp.array([1.0, 2.0, 3.0])
    active_descriptors = jnp.array([0.5, 0.7, 0.9])
    extra_scores = {
        "transitions": QDTransition(
            rewards=jnp.ones((10, 250, 3)),
            passive_desc=jnp.ones((10, 250, 2)),
            state_desc=jnp.ones((10, 250, 4)),
        ),
        "mask": jnp.zeros((10, 250)),
        "full_episode_sols": jnp.zeros(10),
    }
    return active_fitnesses, active_descriptors, extra_scores, random_key


# Define a dummy configuration
@dataclass
class DummyConfig:
    archive: object
    env: object


@dataclass
class ArchiveConfig:
    buffer_length: list


@dataclass
class EnvConfig:
    episode_length: int
    reward_type: str = "full"


# Define a dummy passive behavior descriptor extractor for testing
def passive_behaviour_descriptor_extractor(data, mask):
    mask = jnp.expand_dims(mask, axis=-1)
    descriptors = jnp.sum(data[:, :, :2] * (1.0 - mask), axis=1)
    descriptors = descriptors / jnp.sum(1.0 - mask, axis=1)
    return descriptors


# Passive scoring function to test
def test_random(
    scoring_fn,
    reward_type,
    steps,
    passive_behaviour_descriptor_extractor,
):
    @wraps(scoring_fn)
    def _wrapper(policies_params, random_key):
        active_fitnesses, active_descriptors, extra_scores, random_key = scoring_fn(
            policies_params, random_key
        )

        data = extra_scores["transitions"]
        done_mask = extra_scores["mask"]
        full_episode_sols = extra_scores["full_episode_sols"]

        done_mask_expanded = jnp.stack([done_mask] * len(steps))
        full_episode_sols_expanded = jnp.stack([full_episode_sols] * len(steps))

        # Test 1: Check shapes and values of expanded masks
        assert done_mask_expanded.shape == (
            len(steps),
            10,
            250,
        ), "Invalid shape for done_mask_expanded"
        assert jnp.all(
            done_mask_expanded == 0
        ), "Unexpected non-zero values in done_mask_expanded"

        assert full_episode_sols_expanded.shape == (
            len(steps),
            10,
        ), "Invalid shape for full_episode_sols_expanded"
        assert jnp.all(
            full_episode_sols_expanded == 0
        ), "Unexpected non-zero values in full_episode_sols_expanded"

        mask_to_set = jnp.arange(done_mask.shape[-1]) < jnp.array(steps)[:, None, None]
        masks = jnp.where(mask_to_set, 1.0, done_mask_expanded)

        # Test 2: Check shapes and values of masks
        assert masks.shape == (len(steps), 10, 250), "Invalid shape for masks"
        assert jnp.all(
            masks[0, :, 0 : steps[0]] == 1.0
        ), "Unexpected values in masks (step 0)"
        assert jnp.all(
            masks[0, :, steps[0] : 250] == 0
        ), "Unexpected values in masks (beyond step 0)"

        contact_data = jnp.mean(data.state_desc[:, :, :2], axis=-1)
        contact_data_expanded = jnp.stack([contact_data] * len(steps))
        mask_to_set = jnp.arange(done_mask.shape[-1]) < jnp.array(steps)[:, None, None]
        contact_data_expanded = jnp.where(mask_to_set, 0.0, contact_data_expanded)
        stand_masks = (
            jnp.sum(contact_data_expanded, axis=-1) == 0.0
        )  # true means the robot stands after the buffer time

        # Test 3: Check shapes and values of stand_masks
        assert contact_data_expanded.shape == (len(steps), 10, 250)
        assert jnp.all(contact_data_expanded[0, :, : steps[0]] == 0.0)

        passive_fitnesses = jax.vmap(compute_fitnesses, in_axes=(None, 0, None))(
            data, masks, reward_type
        )
        # Test 4: Check passive fitnesses
        assert passive_fitnesses.shape == (
            len(steps),
            10,
        ), "Invalid shape for passive_fitnesses"

        passive_fitnesses = jnp.where(stand_masks, passive_fitnesses, -jnp.inf)
        passive_fitnesses = jnp.where(
            full_episode_sols_expanded, passive_fitnesses, -jnp.inf
        )
        # Test 5: Check passive fitnesses after setting -inf
        assert jnp.all(
            passive_fitnesses == -jnp.inf
        ), "Unexpected values in passive_fitnesses (step 0)"

        passive_descriptors = jax.vmap(
            passive_behaviour_descriptor_extractor, in_axes=(None, 0)
        )(data.passive_desc, masks)
        # Test 6: Check passive descriptors
        assert passive_descriptors.shape == (
            len(steps),
            10,
            2,
        ), "Invalid shape for passive_descriptors"

        return (
            active_fitnesses,
            active_descriptors,
            {
                "transitions": data,
                "passive_descriptors": None,
                "passive_fitnesses": None,
            },
            random_key,
        )

    return _wrapper


def scoring_fn_custom(policies_params, random_key):
    extra_scores = {
        "transitions": QDTransition(
            rewards=jnp.array(
                [
                    [
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3],
                    ],
                    [
                        [4, 5, 6],
                        [4, 5, 6],
                        [4, 5, 6],
                        [4, 5, 6],
                        [4, 5, 6],
                        [4, 5, 6],
                        [4, 5, 6],
                        [4, 5, 6],
                        [4, 5, 6],
                        [4, 5, 6],
                    ],
                ]
            ),
            passive_desc=jnp.array(
                [
                    [
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                    ],
                    [
                        [0, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                    ],
                ]
            ),
            state_desc=jnp.array(
                [
                    [
                        [0, 1],
                        [1, 0],
                        [0, 0],
                        [0, 0],
                        [0, 1],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                    ],
                    [
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 1],
                    ],
                ]
            ),
        ),
        "mask": jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ]
        ),
        "full_episode_sols": jnp.array([1, 0]),
    }
    return None, None, extra_scores, random_key


def test_custom_values(
    scoring_fn,
    reward_type,
    steps,
    passive_behaviour_descriptor_extractor,
):
    @wraps(scoring_fn)
    def _wrapper(policies_params, random_key):
        active_fitnesses, active_descriptors, extra_scores, random_key = scoring_fn(
            policies_params, random_key
        )

        data = extra_scores["transitions"]
        done_mask = extra_scores["mask"]
        full_episode_sols = extra_scores["full_episode_sols"]

        done_mask_expanded = jnp.stack([done_mask] * len(steps))
        full_episode_sols_expanded = jnp.stack([full_episode_sols] * len(steps))

        # Test 1: Check shapes of expanded masks
        assert done_mask_expanded.shape == (
            len(steps),
            2,
            10,
        ), "Invalid shape for done_mask_expanded"
        assert full_episode_sols_expanded.shape == (
            len(steps),
            2,
        ), "Invalid shape for full_episode_sols_expanded"

        mask_to_set = jnp.arange(done_mask.shape[-1]) < jnp.array(steps)[:, None, None]
        masks = jnp.where(mask_to_set, 1.0, done_mask_expanded)

        # Test 2: Check shapes and values of masks
        assert masks.shape == (len(steps), 2, 10), "Invalid shape for masks"
        expected_masks = jnp.array(
            [
                [
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                ],
                [
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                ],
            ]
        )
        np.testing.assert_array_equal(
            masks, expected_masks, "Unexpected values in masks"
        )

        contact_data = jnp.mean(data.state_desc[:, :, :2], axis=-1)
        contact_data_expanded = jnp.stack([contact_data] * len(steps))
        mask_to_set = jnp.arange(done_mask.shape[-1]) < jnp.array(steps)[:, None, None]
        contact_data_expanded = jnp.where(mask_to_set, 0.0, contact_data_expanded)
        stand_masks = (
            jnp.sum(contact_data_expanded, axis=-1) == 0.0
        )  # true means the robot stands after the buffer time

        # Test 3: Check shapes and values of stand_masks
        assert contact_data_expanded.shape == (len(steps), 2, 10)
        assert stand_masks.shape == (len(steps), 2)
        expected_stand_masks = jnp.array([[1, 0], [0, 0]])
        np.testing.assert_array_equal(
            stand_masks, expected_stand_masks, "Unexpected values in masks"
        )

        passive_fitnesses = jax.vmap(compute_fitnesses, in_axes=(None, 0, None))(
            data, masks, reward_type
        )
        # Test 4: Check passive fitnesses
        assert passive_fitnesses.shape == (
            len(steps),
            2,
        ), "Invalid shape for passive_fitnesses"
        expected_passive_fitnesses = jnp.array([[5, 8], [8, 20]])
        np.testing.assert_array_equal(passive_fitnesses, expected_passive_fitnesses)

        passive_fitnesses = jnp.where(stand_masks, passive_fitnesses, -jnp.inf)
        passive_fitnesses = jnp.where(
            full_episode_sols_expanded, passive_fitnesses, -jnp.inf
        )
        # Test 5: Check passive fitnesses after setting -inf
        assert passive_fitnesses.shape == (len(steps), 2)
        expected_passive_fitnesses = jnp.array([[5, -jnp.inf], [-jnp.inf, -jnp.inf]])
        np.testing.assert_array_equal(passive_fitnesses, expected_passive_fitnesses)

        passive_descriptors = jax.vmap(
            passive_behaviour_descriptor_extractor, in_axes=(None, 0)
        )(data.passive_desc, masks)
        # Test 6: Check passive descriptors
        assert passive_descriptors.shape == (
            len(steps),
            2,
            2,
        ), "Invalid shape for passive_descriptors"
        print(passive_descriptors[0], passive_descriptors[1])
        expected_passive_descriptors = jnp.array(
            [[[0.4, 0.6], [0.5, 0.5]], 
             [[0.5, 0.5], [0.4, 0.6]]]
        )
        np.testing.assert_array_equal(passive_descriptors, expected_passive_descriptors)

        return (
            active_fitnesses,
            active_descriptors,
            {
                "transitions": data,
                "passive_descriptors": None,
                "passive_fitnesses": None,
            },
            random_key,
        )

    return _wrapper


if __name__ == "__main__":
    # TEST WITH RANDOM VALUES - array shapes and values
    cfg = DummyConfig(
        archive=ArchiveConfig(buffer_length=[0.5, 0.6, 0.8]),
        env=EnvConfig(episode_length=250, reward_type="full"),
    )
    steps = tuple(
        [round(s * cfg.env.episode_length) for s in cfg.archive.buffer_length]
    )
    test_random(
        scoring_fn_rdm,
        reward_type="full",
        steps=steps,
        passive_behaviour_descriptor_extractor=passive_behaviour_descriptor_extractor,
    )(None, None)

    # TEST WITH CUSTOM VALUES - check correct computations
    cfg = DummyConfig(
        archive=ArchiveConfig(buffer_length=[0.5, 0.25]),
        env=EnvConfig(episode_length=10, reward_type="full"),
    )
    steps = tuple(
        [round(s * cfg.env.episode_length) for s in cfg.archive.buffer_length]
    )
    test_custom_values(
        scoring_fn_custom,
        reward_type="full",
        steps=steps,
        passive_behaviour_descriptor_extractor=passive_behaviour_descriptor_extractor,
    )(None, None)

    print("All tests passed!")
