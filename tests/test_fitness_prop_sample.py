import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np

from qdax.core.containers.l_value_archive import UnstructuredRepertoire

jax.config.update("jax_disable_jit", True)


class TestFitnessPropSample(unittest.TestCase):
    def setUp(self):
        # Set up test genotypes and fitnesses
        self.genotypes = jnp.array([10, 20, 30, 40, 50])
        self.fitnesses = jnp.array([0.1, 0.2, -jnp.inf, 0.5, 1.0])

        # Set up other mock fields with placeholder data (won't be used in the test)
        self.descriptors = jnp.zeros((5, 2))
        self.passive_descriptors = jnp.zeros((5, 2))
        self.observations = jnp.zeros((5, 3))

        # Create an instance of UnstructuredRepertoire
        self.repertoire = UnstructuredRepertoire(
            genotypes=self.genotypes,
            fitnesses=self.fitnesses,
            descriptors=self.descriptors,
            passive_descriptors=self.passive_descriptors,
            observations=self.observations,
            l_value=0.2,
            max_size=5,
        )

    @patch("jax.random.choice")  # Patch jax.random.choice for deterministic selection
    @patch("jax.random.split")  # Patch jax.random.split to control random key splitting
    def test_fitness_prop_sample(self, mock_split, mock_choice):
        # Set up deterministic splitting of the random key
        random_key = jax.random.PRNGKey(42)

        # Set up deterministic sampling from the mocked genotypes
        # We expect this to be called with probabilities proportional to fitnesses
        # Assign mock output from jax.random.choice to simulate the samples
        mock_choice.side_effect = lambda key, x, shape, p: np.array([50, 40])

        # Call the method under test with the random key and number of samples
        num_samples = 2
        samples = self.repertoire.fitness_prop_sample(random_key, num_samples)

        # Check that jax.random.choice was called with correct probabilities
        fitness_min = 0.1  # The minimum fitness (ignoring -inf)
        shifted_fitnesses = self.fitnesses - fitness_min
        shifted_fitnesses = jnp.where(
            self.fitnesses == -jnp.inf, 0.0, shifted_fitnesses
        )
        expected_prob = shifted_fitnesses / jnp.sum(shifted_fitnesses)

        # Access the arguments passed to jax.random.choice
        choice_args, choice_kwargs = mock_choice.call_args  # (key, x, shape, p)
        # Check that the genotypes and other parameters match
        np.testing.assert_array_equal(choice_args[1], self.genotypes)
        self.assertEqual(choice_kwargs["shape"], (num_samples,))  # Shape
        np.testing.assert_allclose(choice_kwargs["p"], expected_prob, rtol=1e-6)
        # Check that the returned samples are as expected
        np.testing.assert_array_equal(samples, np.array([50, 40]))


if __name__ == "__main__":
    unittest.main()
