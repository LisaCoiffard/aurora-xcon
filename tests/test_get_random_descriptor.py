from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tasks.bd_extractors import get_random_descriptor


class QDTransition(NamedTuple):
    obs: jnp.ndarray


def test_get_random_descriptor():
    batch_size, traj_length, obs_dim = 3, 4, 6
    data = jnp.array(
        [
            [
                [1, 0, 1, 0, 0.5, 0.8],
                [1, 1, 0, 1, 0.6, 0.9],
                [0, 1, 1, 0, 0.7, 1.0],
                [0, 0, 1, 1, 0.8, 0.7],
            ],  # Traj 1
            [
                [0, 1, 0, 1, 0.2, 0.3],
                [1, 0, 1, 0, 0.3, 0.4],
                [1, 1, 0, 1, 0.4, 0.5],
                [0, 0, 0, 0, 0.5, 0.6],
            ],  # Traj 2
            [
                [1, 1, 1, 0, 0.1, 0.2],
                [0, 0, 1, 1, 0.2, 0.3],
                [0, 0, 0, 0, 0.3, 0.4],
                [0, 0, 0, 0, 0.4, 0.5],
            ],  # Traj 3
        ]
    )

    mask = jnp.array(
        [
            [0, 0, 0, 1],  # Traj 1 ends at t=3
            [0, 0, 1, 1],  # Traj 2 ends at t=2
            [0, 1, 1, 1],  # Traj 3 ends at t=1
        ]
    )

    # Test binary and non-binary descriptors
    bd_dims = (0, 4)  # First (binary) and fifth (non-binary) dimensions
    is_binary = jnp.array([True, False])
    min_bd = jnp.array([0.0, 0.0])
    max_bd = jnp.array([1.0, 1.0])

    result = get_random_descriptor(
        QDTransition(data), mask, bd_dims, is_binary, min_bd, max_bd
    )

    # Expected: First dim is proportion of 1s, second is last unmasked value
    expected = jnp.array(
        [
            [2/3, 0.7],  # Traj 1: 2/3 ones until t=3, value 0.7 at t=2
            [0.5, 0.3],  # Traj 2: 1/2 ones until t=2, value 0.3 at t=1
            [1.0, 0.1],  # Traj 3: 1 ones until t=1, value 0.1 at t=0
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-7)


if __name__ == "__main__":
    test_get_random_descriptor()
    print("All tests passed.")
