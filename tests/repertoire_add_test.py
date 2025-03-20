import numpy as np
import jax
import jax.numpy as jnp
from qdax.custom_types import (
    Genotype,
    Descriptor,
    Fitness,
    Observation,
)
from qdax.core.containers.l_value_archive import (
    get_cells_indices,
    intra_batch_comp,
)


# Dummy self class to simulate the necessary state
class SelfClass:
    def __init__(self, random_key=None):
        # self.fitnesses = jnp.arange(10, dtype=jnp.float32)
        self.fitnesses = jnp.array(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -jnp.inf, 8.0, 9.0]
        )
        # self.fitnesses = self.fitnesses.at[5].set(-jnp.inf)
        # self.fitnesses = self.fitnesses.at[7].set(-jnp.inf)
        # self.descriptors = jax.random.uniform(
        #     random_key, (10, 2), minval=0, maxval=1, dtype=jnp.float32
        # )
        self.descriptors = jnp.array(
            [
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.3],
                [0.4, 0.4],
                [0.5, 0.5],
                [0.6, 0.6],
                [0.7, 0.7],
                [0.8, 0.8],
                [0.9, 0.9],
                [1.0, 1.0],
            ]
        )

        self.l_value = 0.1
        self.max_size = 10

    def add(
        self,
        batch_of_descriptors,
        batch_of_fitnesses,
    ):
        # We need to replace all the descriptors that are not filled with jnp inf
        filtered_descriptors = jnp.where(
            jnp.expand_dims((self.fitnesses == -jnp.inf), axis=-1),
            jnp.full(self.descriptors.shape[-1], fill_value=jnp.inf),
            self.descriptors,
        )

        # Get indices of the 2 nearest neighbours in the repertoire compared to batch
        batch_of_indices, batch_of_distances = get_cells_indices(
            batch_of_descriptors, filtered_descriptors, 2
        )

        # Save the second-nearest neighbours to check a condition
        second_neighbours = batch_of_distances.at[..., 1].get()

        # Keep the Nearest neighbours
        batch_of_indices = batch_of_indices.at[..., 0].get()

        # Keep the Nearest neighbours
        batch_of_distances = batch_of_distances.at[..., 0].get()

        # We remove individuals that are too close to the second nn.
        # This avoids having clusters of individuals after adding them.
        not_novel_enough = jnp.where(
            jnp.squeeze(second_neighbours <= self.l_value), True, False
        )

        # batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

        # TODO: Doesn't Work if Archive is full. Need to use the closest individuals
        # in that case.
        empty_indexes = jnp.squeeze(
            jnp.nonzero(
                jnp.where(jnp.isinf(self.fitnesses), 1, 0),
                size=batch_of_indices.shape[0],
                fill_value=-1,
            )[0]
        )
        batch_of_indices = jnp.where(
            jnp.squeeze(batch_of_distances <= self.l_value),
            jnp.squeeze(batch_of_indices),
            -1,
        )

        # We get all the indices of the empty bds first and then the filled ones
        # (because of -1)
        sorted_bds = jax.lax.top_k(
            -1 * batch_of_indices.squeeze(), batch_of_indices.shape[0]
        )[1]
        batch_of_indices = jnp.where(
            jnp.squeeze(batch_of_distances.at[sorted_bds].get() <= self.l_value),
            batch_of_indices.at[sorted_bds].get(),
            empty_indexes,
        )

        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        # ReIndexing of all the inputs to the correct sorted way
        batch_of_descriptors = batch_of_descriptors.at[sorted_bds].get()
        batch_of_fitnesses = batch_of_fitnesses.at[sorted_bds].get()
        not_novel_enough = not_novel_enough.at[sorted_bds].get()

        # Check to find Individuals with same BD within the Batch
        keep_indiv = jax.jit(
            jax.vmap(intra_batch_comp, in_axes=(0, 0, None, None, None), out_axes=(0))
        )(
            batch_of_descriptors.squeeze(),
            jnp.arange(
                0, batch_of_descriptors.shape[0], 1
            ),  # keep track of where we are in the batch to assure right comparisons
            batch_of_descriptors.squeeze(),
            batch_of_fitnesses.squeeze(),
            self.l_value,
        )

        keep_indiv = jnp.logical_and(keep_indiv, jnp.logical_not(not_novel_enough))

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(),
            num_segments=self.max_size,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
        )

        # get addition condition
        grid_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(grid_fitnesses, batch_of_indices, 0)
        addition_condition = batch_of_fitnesses > current_fitnesses
        addition_condition = jnp.logical_and(
            addition_condition, jnp.expand_dims(keep_indiv, axis=-1)
        )

        # assign fake position when relevant : num_centroids is out of bounds
        batch_of_indices = jnp.where(
            addition_condition,
            batch_of_indices,
            self.max_size,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze()].set(
            batch_of_fitnesses.squeeze()
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze()].set(
            batch_of_descriptors.squeeze()
        )

        return new_descriptors, new_fitnesses


def test_add():
    random_key = jax.random.PRNGKey(0)
    random_key, subkey = jax.random.split(random_key)
    # Create dummy inputs
    # batch_of_fitnesses = jnp.arange(3, dtype=jnp.float32)
    # batch_of_fitnesses = batch_of_fitnesses.at[0].set(-jnp.inf)
    # batch_of_descriptors = jax.random.uniform(
    #     random_key, (3, 2), minval=0, maxval=1, dtype=jnp.float32
    # )
    batch_of_fitnesses = jnp.array([0.0, 0.1, 4.0])
    batch_of_descriptors = jnp.array([[0.6, 0.6], [0.8, 0.8], [0.3, 0.3]])

    # Instantiate self and call the add method
    random_key, subkey = jax.random.split(random_key)
    self_instance = SelfClass(random_key=subkey)
    new_descriptors, new_fitnesses = self_instance.add(
        batch_of_descriptors,
        batch_of_fitnesses,
    )

    # Print result
    print("New Fitnesses:", new_fitnesses)
    print("New Descriptors:", new_descriptors)


if __name__ == "__main__":
    test_add()
