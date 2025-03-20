import jax.numpy as jnp
import numpy as np

from tasks.scoring import (
    compute_fitnesses,
    terminate_on_contact,
)


class DummyData:
    pass


def test_compute_fitnesses():
    # create data variable, a dummy array of shape (2, 3, 3) with random but fixed low values
    data = DummyData()
    data.rewards = np.array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    )
    mask = np.array([[0, 0, 0], [0, 1, 1]])
    reward_type = "full"

    result = compute_fitnesses(data, mask, reward_type)
    expected_result = np.array([12, 1])
    np.testing.assert_array_equal(result, expected_result)

    reward_type = "no_forward"

    result = compute_fitnesses(data, mask, reward_type)
    expected_result = np.array([15, 2])
    np.testing.assert_array_equal(result, expected_result)

    reward_type = "forward_only"

    result = compute_fitnesses(data, mask, reward_type)
    expected_result = np.array([18, 3])
    np.testing.assert_array_equal(result, expected_result)


def test_terminate_on_contact():
    data = DummyData()

    # TEST 1
    data.state_desc = jnp.array(
        [
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    buffer = 0.5
    step = int(data.state_desc.shape[1] * buffer)

    result = terminate_on_contact(
        data=data,
        step=step,
    )
    expected_result = jnp.array([True, False, True, False])
    np.testing.assert_array_equal(result, expected_result)

    # TEST 2
    data.state_desc = jnp.array(
        [
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    buffer = 0.75
    step = int(data.state_desc.shape[1] * buffer)

    result = terminate_on_contact(
        data=data,
        step=step,
    )
    expected_result = jnp.array([True, True, True, True])
    np.testing.assert_array_equal(result, expected_result)

    # TEST 3
    data.state_desc = jnp.array(
        [
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    buffer = 0.75
    step = int(data.state_desc.shape[1] * buffer)

    result = terminate_on_contact(
        data=data,
        step=step,
    )
    expected_result = jnp.array([False, True])
    np.testing.assert_array_equal(result, expected_result)


def test_remove_full_episode_sols():
    episode_length = 250
    buffer_length = 0.2

    # check step is computed correctly
    step = int(episode_length * buffer_length)
    assert step == 50

    mask = jnp.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 1]])
    fitnesses = jnp.array([1, 2, 3])
    step = 2

    # check mask for full epidsode solutions is correct
    full_episode_sols = jnp.sum(mask, axis=1) == 0.0
    expected_full_episode_sols = jnp.array([1, 0, 0])
    np.testing.assert_array_equal(full_episode_sols, expected_full_episode_sols)

    # check mask is filled up until the buffer ends
    mask = mask.at[:, :step].set(1.0)
    expected_mask = jnp.array([[1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]])
    np.testing.assert_array_equal(mask, expected_mask)

    # check the correct solutions are kept
    keep_fitnesses = jnp.where(full_episode_sols, fitnesses, -jnp.inf)
    expected_keep_fitnesses = jnp.array([1.0, -jnp.inf, -jnp.inf])
    np.testing.assert_array_equal(keep_fitnesses, expected_keep_fitnesses)


if __name__ == "__main__":
    test_terminate_on_contact()
    test_compute_fitnesses()
    test_remove_full_episode_sols()
    print("All tests passed!")
