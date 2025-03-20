import jax.numpy as jnp
import numpy as np

from tasks.bd_extractors import (
    get_standing_back_feet,
    get_standing_back_feet_z_pos,
    get_standing_back_feet_z_pos_rot,
    get_standing_prop,
    get_standing_z_pos,
    get_standing_z_rot,
    get_feet_contact_proportion,
    get_z_position,
    get_z_rotation,
    get_leg_standing_prop,
    get_distance_from_init_pos,
)


def test_get_distance_desc():
    data = jnp.array(
        [
            [
                [0.0, 1.0, 0.0, 1.0, 0.3, 0.0, -15, -15],
                [0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 15, 15],
            ],
            [
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57, 0, 0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57, 5, -3],
            ],
        ]
    )
    mask = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    expected_result = jnp.array([[1], [0.1374368542]])
    result = get_distance_from_init_pos(data, mask)
    np.testing.assert_array_almost_equal(result, expected_result)


def test_get_standing_prop():

    # TEST 1
    data = jnp.array(
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

    result = get_standing_prop(
        data=data,
        mask=None,
    )
    expected_result = jnp.array([[0.75], [0.25], [0.75], [0.25]])
    np.testing.assert_array_almost_equal(result, expected_result)

    # TEST 2
    data = jnp.array(
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
        ]
    )

    result = get_standing_prop(
        data=data,
        mask=None,
    )
    expected_result = jnp.array([[0.75], [0.25], [0.25]])
    np.testing.assert_array_almost_equal(result, expected_result)

    # TEST 3
    data = jnp.array(
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

    result = get_standing_prop(
        data=data,
        mask=None,
    )
    expected_result = jnp.array([[0.0], [0.25]])
    np.testing.assert_array_almost_equal(result, expected_result)

    # TEST 4 - repeat is True
    data = jnp.array(
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

    result = get_standing_prop(
        data=data,
        mask=None,
        repeat=True,
    )
    expected_result = jnp.array([[0.0, 0.0], [0.25, 0.25]])
    np.testing.assert_array_almost_equal(result, expected_result)


def test_get_feet_contact_proportion():

    # TEST 1 - different feet names
    data = jnp.array(
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
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    mask = jnp.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # individual survives whole episode
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
            ],  # individual dies before episode ends
        ]
    )
    feet_name = "front"

    result = get_feet_contact_proportion(
        data=data,
        mask=mask,
        feet_name=feet_name,
    )
    expected_result = jnp.array(
        [
            [0.5, 0.375],
            [0.2, 0.4],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 2))

    feet_name = "back"

    result = get_feet_contact_proportion(
        data=data,
        mask=mask,
        feet_name=feet_name,
    )
    expected_result = jnp.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 2))

    feet_name = "all"

    result = get_feet_contact_proportion(
        data=data,
        mask=mask,
        feet_name=feet_name,
    )
    expected_result = jnp.array(
        [
            [0.5, 0.375, 0.0, 1.0],
            [0.2, 0.4, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 4))

    # TEST 2 - different masks
    data = jnp.array(
        [
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    mask = jnp.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # individual survives whole episode
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ],  # individual dies before episode ends + buffer for first step
            [
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # individual survives whole episode + buffer for first 3 steps
        ]
    )
    feet_name = "all"

    result = get_feet_contact_proportion(
        data=data,
        mask=mask,
        feet_name=feet_name,
    )
    expected_result = jnp.array(
        [
            [0.5, 0.375, 0.25, 1.0],
            [0.4, 0.2, 0.0, 0.8],
            [0.4, 0.2, 0.2, 0.4],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 4))


def test_get_z_position():

    # TEST 1 - max/mean values
    data = jnp.array(
        [
            [
                [0.0, 1.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 1.0, 4.0, 1.57],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
            ],
            [
                [0.0, 1.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
            ],
        ]
    )
    mask = jnp.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # individual survives whole episode
            [
                0.0,
                1.0,
                1.0,
                1.0,
            ],  # individual dies before episode ends
        ]
    )
    val_type = "max"

    result = get_z_position(data=data, mask=mask, val_type=val_type)
    expected_result = jnp.array(
        [
            [1.0],
            [0.0],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 1))

    val_type = "mean"

    result = get_z_position(data=data, mask=mask, val_type=val_type)
    expected_result = jnp.array(
        [
            [0.25],
            [0.0],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 1))


def test_get_z_rotation():

    # TEST 1 - max/mean values
    data = jnp.array(
        [
            [
                [0.0, 1.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 1.0, 4.0, 1.57],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 1.0, 4.0, 1.57],
            ],
            [
                [0.0, 1.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
            ],
        ]
    )
    mask = jnp.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # individual survives whole episode
            [
                0.0,
                1.0,
                1.0,
                1.0,
            ],  # individual dies before episode ends
        ]
    )
    val_type = "max"

    result = get_z_rotation(data=data, mask=mask, val_type=val_type)
    expected_result = jnp.array(
        [
            [1.0],
            [0.0],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 1))

    val_type = "mean"

    result = get_z_rotation(data=data, mask=mask, val_type=val_type)
    expected_result = jnp.array(
        [
            [1.0],
            [0.0],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 1))


def test_bd_combinations():
    data = jnp.array(
        [
            [
                [0.0, 1.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 1.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 0.0, 4.0, 1.57],
                [0.0, 0.0, 0.0, 0.0, 0.3, 1.57],
                [1.0, 0.0, 0.0, 1.0, 0.3, 1.57],
            ],
            [
                [0.0, 1.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 1.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 0.0, 4.0, 1.57],
                [0.0, 0.0, 0.0, 0.0, 0.3, 1.57],
                [1.0, 0.0, 0.0, 1.0, 0.3, 1.57],
            ],
            [
                [0.0, 1.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 1.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 0.0, 4.0, 1.57],
                [0.0, 0.0, 0.0, 0.0, 0.3, 1.57],
                [1.0, 0.0, 0.0, 1.0, 0.3, 1.57],
            ],
            [
                [0.0, 1.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 1.0, 0.0, 1.0, 0.3, 1.57],
                [0.0, 0.0, 0.0, 0.0, 4.0, 1.57],
                [0.0, 0.0, 0.0, 0.0, 0.3, 1.57],
                [1.0, 0.0, 0.0, 1.0, 0.3, 1.57],
            ],
        ]
    )
    mask = jnp.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # individual survives whole episode
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
            ],  # individual dies before episode ends
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ],  # individual dies before episode ends + buffer for first step
            [
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # individual survives whole episode + buffer for first 3 steps
        ]
    )
    result = get_standing_back_feet(
        data=data,
        mask=mask,
        feet_name="back",
    )
    expected_result = jnp.array(
        [[0.0, 0.75, 0.0], [0.0, 1.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.6, 0.0]]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 3))

    result = get_standing_z_pos(
        data=data,
        mask=mask,
        val_type="max",
    )
    expected_result = jnp.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 2))

    result = get_standing_z_rot(
        data=data,
        mask=mask,
        val_type="max",
    )
    expected_result = jnp.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 2))

    result = get_standing_back_feet_z_pos_rot(
        data=data,
        mask=mask,
        feet_name="back",
        val_type="max",
    )
    expected_result = jnp.array(
        [
            [0.0, 0.75, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.8, 0.0, 1.0, 1.0],
            [0.0, 0.6, 0.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 5))

    result = get_standing_back_feet_z_pos(
        data=data,
        mask=mask,
        feet_name="back",
        val_type="max",
    )
    expected_result = jnp.array(
        [
            [0.0, 0.75, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 1.0],
            [0.0, 0.6, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)
    np.testing.assert_equal(result.shape, (data.shape[0], 4))


def test_get_leg_standing_prop():
    # TEST 1
    data = jnp.array(
        [
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    result = get_leg_standing_prop(
        data=data,
        mask=None,
    )
    expected_result = jnp.array([[1, 0.8, 1, 0], [1, 0.4, 1, 0], [0.8, 0.8, 1, 0]])
    np.testing.assert_array_almost_equal(result, expected_result)


if __name__ == "__main__":
    test_get_standing_prop()
    test_get_feet_contact_proportion()
    test_get_z_position()
    test_get_z_rotation()
    # test_bd_combinations()
    test_get_leg_standing_prop()
    test_get_distance_desc()
    print("All tests passed!")
