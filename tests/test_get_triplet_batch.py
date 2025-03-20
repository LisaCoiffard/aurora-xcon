import unittest
import jax
import jax.numpy as jnp
import numpy as np
from ae_utils.model_train import get_triplet_batch
from qdax.core.containers.l_value_archive import UnstructuredRepertoire


class TestGetTripletBatch(unittest.TestCase):

    def setUp(self):
        self.repertoire_size = 100

    def generate_random_repertoire(self):
        random_key = jax.random.PRNGKey(0)

        # Generate random genotypes and fitnesses
        self.genotypes = jax.random.normal(
            jax.random.PRNGKey(0), (self.repertoire_size, 10)
        )
        self.fitnesses = jax.random.uniform(
            jax.random.PRNGKey(1), (self.repertoire_size,)
        )

        # Other fields can be filled with placeholder values
        self.descriptors = jax.random.normal(
            jax.random.PRNGKey(2), (self.repertoire_size, 2)
        )
        self.passive_descriptors = jax.random.normal(
            jax.random.PRNGKey(3), (self.repertoire_size, 2)
        )
        self.observations = jax.random.normal(
            jax.random.PRNGKey(4), (self.repertoire_size, 3)
        )

        # Create an instance of UnstructuredRepertoire
        self.repertoire = UnstructuredRepertoire(
            genotypes=self.genotypes,
            fitnesses=self.fitnesses,
            descriptors=self.descriptors,
            passive_descriptors=self.passive_descriptors,
            observations=self.observations,
            l_value=0.2,
            max_size=self.repertoire_size,
        )

    def test_get_triplet_batch(self):
        num_tests = 10  # Number of tests

        for i in range(num_tests):
            print(f"Running test {i+1}/{num_tests}")
            # Generate a new random repertoire for each test
            repertoire = self.generate_random_repertoire()

            # Use all repertoire indices for the batch (0 to 99)
            batch_indices = jnp.arange(self.repertoire.fitnesses.size)

            # Generate a random key
            random_key = jax.random.PRNGKey(i)  # Different key for each test

            # Call get_triplet_batch
            triplets = get_triplet_batch(self.repertoire, batch_indices, random_key)

            # Iterate through each triplet and validate the conditions
            for triplet in triplets:
                anchor_idx, pos_idx, neg_idx = triplet

                # Ensure positive and negative samples are not the same
                self.assertNotEqual(
                    pos_idx, neg_idx, "Positive and negative samples are the same."
                )

                # Ensure positive and negative samples are not the same as the anchor
                self.assertNotEqual(
                    anchor_idx, pos_idx, "Positive sample is the same as the anchor."
                )
                self.assertNotEqual(
                    anchor_idx, neg_idx, "Negative sample is the same as the anchor."
                )

                # Check that positive is closer to the anchor in terms of fitness than the negative
                anchor_fitness = self.repertoire.fitnesses[anchor_idx]
                pos_fitness = self.repertoire.fitnesses[pos_idx]
                neg_fitness = self.repertoire.fitnesses[neg_idx]

                pos_distance = jnp.abs(anchor_fitness - pos_fitness)
                neg_distance = jnp.abs(anchor_fitness - neg_fitness)

                self.assertLess(
                    pos_distance,
                    neg_distance,
                    "Positive sample is not closer than negative sample.",
                )


if __name__ == "__main__":
    unittest.main()
