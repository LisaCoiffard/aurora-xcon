from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qdax.custom_types import Descriptor, Fitness, Genotype, Observation, RNGKey


class UnstructuredRepertoire(flax.struct.PyTreeNode):
    """
    Class for the unstructured repertoire in Map Elites.

    Args:
            genotypes: a PyTree containing all the genotypes in the repertoire ordered
                    by the centroids. Each leaf has a shape (num_centroids, num_features). The
                    PyTree can be a simple Jax array or a more complex nested structure such
                    as to represent parameters of neural network in Flax.
            fitnesses: an array that contains the fitness of solutions in each cell of the
                    repertoire, ordered by centroids. The array shape is (num_centroids,).
            descriptors: an array that contains the descriptors of solutions in each cell
                    of the repertoire, ordered by centroids. The array shape
                    is (num_centroids, num_descriptors).
            observations: observations that the genotype gathered in the environment.
    """

    genotypes: Genotype
    fitnesses: Fitness
    descriptors: Descriptor
    observations: Observation
    max_size: int = flax.struct.field(pytree_node=False)
    d_min: jnp.ndarray

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_observations: Observation,
    ) -> UnstructuredRepertoire:
        """Adds a batch of genotypes to the repertoire.

        Args:
                batch_of_genotypes: genotypes of the individuals to be considered
                        for addition in the repertoire.
                batch_of_descriptors: associated descriptors.
                batch_of_fitnesses: associated fitness.
                batch_of_observations: associated observations.

        Returns:
                A new unstructured repertoire where the relevant individuals have been
                added.
        """
        # Concatenate everything
        genotypes = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            self.genotypes,
            batch_of_genotypes,
        )
        descriptors = jnp.concatenate([self.descriptors, batch_of_descriptors], axis=0)

        fitnesses = jnp.concatenate([self.fitnesses, batch_of_fitnesses], axis=0)
        observations = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            self.observations,
            batch_of_observations,
        )

        is_empty = fitnesses == -jnp.inf

        # Fitter
        fitter = fitnesses[:, None] < fitnesses[None, :]
        fitter = jnp.where(
            is_empty[None, :], False, fitter
        )  # empty individuals can not be fitter
        fitter = jnp.fill_diagonal(
            fitter, False, inplace=False
        )  # an individual can not be fitter than itself

        # Distance to k-fitter-nearest neighbors
        distance = jnp.linalg.norm(
            descriptors[:, None, :] - descriptors[None, :, :], axis=-1
        )
        distance = jnp.where(fitter, distance, jnp.inf)
        values, indices = jax.vmap(partial(jax.lax.top_k, k=3))(-distance)
        distance = jnp.mean(
            -values, where=jnp.take_along_axis(fitter, indices, axis=1), axis=-1
        )  # if number of fitter individuals is less than k, top_k will return at least one inf
        distance = jnp.where(
            jnp.isnan(distance), jnp.inf, distance
        )  # if no individual is fitter, set distance to inf
        distance = jnp.where(
            is_empty, -jnp.inf, distance
        )  # empty cells have distance -inf

        # Sort by distance to k-fitter-nearest neighbors
        indices = jnp.argsort(distance, descending=True)
        d_min = (jnp.sort(distance, descending=True)[: self.max_size])[-1]

        indices = indices[: self.max_size]
        is_offspring_added = jax.vmap(lambda i: jnp.any(indices == i))(
            jnp.arange(self.max_size, self.max_size + batch_of_fitnesses.size)
        )

        # Sort
        genotypes = jax.tree.map(lambda x: x[indices], genotypes)
        descriptors = descriptors[indices]
        fitnesses = fitnesses[indices]
        observations = jax.tree.map(lambda x: x[indices], observations)

        return (
            UnstructuredRepertoire(
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                observations=observations,
                max_size=self.max_size,
                d_min=d_min,
            ),
            is_offspring_added,
        )

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """Sample elements in the repertoire.

        Args:
                random_key: a jax PRNG random key
                num_samples: the number of elements to be sampled

        Returns:
                samples: a batch of genotypes sampled in the repertoire
                random_key: an updated jax PRNG random key
        """

        random_key, sub_key = jax.random.split(random_key)
        grid_empty = self.fitnesses == -jnp.inf
        p = (1.0 - grid_empty) / jnp.sum(1.0 - grid_empty)

        samples = jax.tree.map(
            lambda x: jax.random.choice(sub_key, x, shape=(num_samples,), p=p),
            self.genotypes,
        )

        return samples, random_key

    @partial(jax.jit, static_argnames=("num_samples",))
    def fitness_prop_sample(
        self, random_key: RNGKey, num_samples: int
    ) -> Tuple[Genotype, RNGKey]:
        """Sample elements in the repertoire with a fitness proportional sampling.

        Args:
                random_key: a jax PRNG random key
                num_samples: the number of elements to be sampled

        Returns:
                samples: a batch of genotypes sampled in the repertoire
                random_key: an updated jax PRNG random key
        """

        # Shift fitnesses to avoid negative values
        fitness_min = jnp.min(
            jnp.where(self.fitnesses == -jnp.inf, jnp.inf, self.fitnesses)
        )
        fitnesses = self.fitnesses - fitness_min

        grid_empty = self.fitnesses == -jnp.inf
        fitnesses_sum = jnp.sum(jnp.where(grid_empty, 0.0, fitnesses))
        # Compute proportionate probabilities
        prob = fitnesses / fitnesses_sum
        # Handle the case where the grid is empty
        prob = jnp.where(grid_empty, 0.0, prob)

        samples = jax.tree.map(
            lambda x: jax.random.choice(random_key, x, shape=(num_samples,), p=prob),
            self.genotypes,
        )

        return samples

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        observations: Observation,
        max_size: int,
        d_min: jnp.ndarray,
    ) -> UnstructuredRepertoire:
        """Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
                genotypes: initial genotypes, pytree in which leaves
                        have shape (batch_size, num_features)
                fitnesses: fitness of the initial genotypes of shape (batch_size,)
                descriptors: descriptors of the initial genotypes
                        of shape (batch_size, num_descriptors)
                observations: observations experienced in the evaluation task.
                size: size of the repertoire

        Returns:
                an initialized unstructured repertoire.
        """

        # Init repertoire with dummy values
        dummy_genotypes = jax.tree.map(
            lambda x: jnp.full((max_size,) + x.shape[1:], fill_value=jnp.nan),
            genotypes,
        )
        dummy_fitnesses = jnp.full((max_size,), fill_value=-jnp.inf)
        dummy_descriptors = jnp.full(
            (max_size,) + descriptors.shape[1:], fill_value=jnp.nan
        )
        dummy_observations = jax.tree.map(
            lambda x: jnp.full((max_size,) + x.shape[1:], fill_value=jnp.nan),
            observations,
        )

        repertoire = UnstructuredRepertoire(
            genotypes=dummy_genotypes,
            fitnesses=dummy_fitnesses,
            descriptors=dummy_descriptors,
            observations=dummy_observations,
            max_size=max_size,
            d_min=d_min,
        )

        repertoire, is_offspring_added = repertoire.add(
            genotypes,
            descriptors,
            fitnesses,
            observations,
        )
        return repertoire

    def save(self, path: str = "./") -> None:
        """Saves the grid on disk in the form of .npy files.

        Flattens the genotypes to store it with .npy format. Supposes that
        a user will have access to the reconstruction function when loading
        the genotypes.

        Args:
                path: Path where the data will be saved. Defaults to "./".
        """

        def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
            flatten_genotype, _unravel_pytree = ravel_pytree(genotype)
            return flatten_genotype

        # flatten all the genotypes
        flat_genotypes = jax.vmap(flatten_genotype)(self.genotypes)

        # save data
        jnp.save(path + "genotypes.npy", flat_genotypes)
        jnp.save(path + "fitnesses.npy", self.fitnesses)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "observations.npy", self.observations)
        jnp.save(path + "d_min.npy", self.d_min)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> UnstructuredRepertoire:
        """Loads an unstructured repertoire.

        Args:
                reconstruction_fn: Function to reconstruct a PyTree
                        from a flat array.
                path: Path where the data is saved. Defaults to "./".

        Returns:
                An unstructured repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)
        fitnesses = jnp.load(path + "fitnesses.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        observations = jnp.load(path + "observations.npy")
        d_min = jnp.load(path + "d_min.npy")

        return UnstructuredRepertoire(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            observations=observations,
            max_size=fitnesses.size,
            d_min=d_min,
        )

    def extinction(self, remaining_prop: float, random_key) -> UnstructuredRepertoire:
        """Randomly removes a proportion of individuals from the repertoire.

        Args:
            remaining_prop: proportion of individuals to keep (between 0 and 1)

        Returns:
            A new repertoire with only the remaining individuals
        """

        best_index = jnp.argmax(self.fitnesses)

        # Calculate how many individuals to keep (subtracts 1 for the best individual)
        remaining_count = jnp.maximum(
            1, jnp.floor(self.max_size * remaining_prop).astype(jnp.int32)
        )

        # Generate random mask for valid entries, excluding the best index
        mask = jnp.ones(self.max_size, dtype=bool)
        mask = mask.at[best_index].set(False)  # Remove best_idx from consideration
        available_indices = jnp.where(mask)[0]

        # Shuffle available indices and select remaining_count - 1 of them
        shuffled_available = jax.random.permutation(random_key, available_indices)
        selected_random_indices = shuffled_available[: remaining_count - 1]

        # Combine best_idx with randomly selected indices
        selected_indices = jnp.concatenate(
            [jnp.array([best_index]), selected_random_indices]
        )

        # Create mask for selected entries
        final_mask = jnp.zeros(self.max_size, dtype=bool).at[selected_indices].set(True)

        # Create dummy values for non-selected entries
        dummy_fitnesses = jnp.full_like(self.fitnesses, fill_value=-jnp.inf)
        dummy_descriptors = jnp.full_like(self.descriptors, fill_value=jnp.nan)

        # Update repertoire
        new_fitnesses = jnp.where(final_mask, self.fitnesses, dummy_fitnesses)
        new_descriptors = jnp.where(
            final_mask[..., None], self.descriptors, dummy_descriptors
        )

        def mask_genotype(x):
            # Expand mask dimensions to match the genotype shape
            expanded_mask = final_mask
            for _ in range(len(x.shape) - 1):
                expanded_mask = expanded_mask[..., None]
            return jnp.where(expanded_mask, x, jnp.full_like(x, jnp.nan))

        new_genotypes = jax.tree.map(mask_genotype, self.genotypes)
        new_observations = jax.tree.map(mask_genotype, self.observations)

        return UnstructuredRepertoire(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            observations=new_observations,
            max_size=self.max_size,
            d_min=self.d_min,
        )


class UnstructuredRepertoirePassiveDesc(UnstructuredRepertoire):
    """
    Extended version of UnstructuredRepertoire that includes passive descriptors.

    Args:
            passive_descriptors: an array that contains the hand-coded descriptors of 
                    solutions in each cell of the repertoire, ordered by centroids. 
                    The array shape is (num_centroids, num_passive_descriptors).
    """

    passive_descriptors: Descriptor

    def save(self, path: str = "./") -> None:
        """Saves the grid on disk in the form of .npy files.

        Flattens the genotypes to store it with .npy format. Supposes that
        a user will have access to the reconstruction function when loading
        the genotypes.

        Args:
                path: Path where the data will be saved. Defaults to "./".
        """

        def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
            flatten_genotype, _unravel_pytree = ravel_pytree(genotype)
            return flatten_genotype

        # flatten all the genotypes
        flat_genotypes = jax.vmap(flatten_genotype)(self.genotypes)

        # save data
        jnp.save(path + "genotypes.npy", flat_genotypes)
        jnp.save(path + "fitnesses.npy", self.fitnesses)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "observations.npy", self.observations)
        jnp.save(path + "d_min.npy", self.d_min)
        jnp.save(path + "passive_descriptors.npy", self.passive_descriptors)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> UnstructuredRepertoire:
        """Loads an unstructured repertoire.

        Args:
                reconstruction_fn: Function to reconstruct a PyTree
                        from a flat array.
                path: Path where the data is saved. Defaults to "./".

        Returns:
                An unstructured repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)
        fitnesses = jnp.load(path + "fitnesses.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        observations = jnp.load(path + "observations.npy")
        d_min = jnp.load(path + "d_min.npy")
        passive_descriptors = jnp.load(path + "passive_descriptors.npy")

        return UnstructuredRepertoirePassiveDesc(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            passive_descriptors=passive_descriptors,
            observations=observations,
            max_size=fitnesses.size,
            d_min=d_min,
        )

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_passive_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_observations: Observation,
    ) -> UnstructuredRepertoire:
        """Adds a batch of genotypes to the repertoire.

        Args:
                batch_of_genotypes: genotypes of the individuals to be considered
                        for addition in the repertoire.
                batch_of_descriptors: associated descriptors.
                batch_of_passive_descriptors: associated passive descriptors.
                batch_of_fitnesses: associated fitness.
                batch_of_observations: associated observations.

        Returns:
                A new unstructured repertoire where the relevant individuals have been
                added.
        """
        # Concatenate everything
        genotypes = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            self.genotypes,
            batch_of_genotypes,
        )
        descriptors = jnp.concatenate([self.descriptors, batch_of_descriptors], axis=0)

        passive_descriptors = jnp.concatenate(
            [self.passive_descriptors, batch_of_passive_descriptors], axis=0
        )

        fitnesses = jnp.concatenate([self.fitnesses, batch_of_fitnesses], axis=0)
        observations = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            self.observations,
            batch_of_observations,
        )

        is_empty = fitnesses == -jnp.inf

        # Fitter
        fitter = fitnesses[:, None] < fitnesses[None, :]
        fitter = jnp.where(
            is_empty[None, :], False, fitter
        )  # empty individuals can not be fitter
        fitter = jnp.fill_diagonal(
            fitter, False, inplace=False
        )  # an individual can not be fitter than itself

        # Distance to k-fitter-nearest neighbors
        distance = jnp.linalg.norm(
            descriptors[:, None, :] - descriptors[None, :, :], axis=-1
        )
        distance = jnp.where(fitter, distance, jnp.inf)
        values, indices = jax.vmap(partial(jax.lax.top_k, k=3))(-distance)
        distance = jnp.mean(
            -values, where=jnp.take_along_axis(fitter, indices, axis=1), axis=-1
        )  # if number of fitter individuals is less than k, top_k will return at least one inf
        distance = jnp.where(
            jnp.isnan(distance), jnp.inf, distance
        )  # if no individual is fitter, set distance to inf
        distance = jnp.where(
            is_empty, -jnp.inf, distance
        )  # empty cells have distance -inf

        # Sort by distance to k-fitter-nearest neighbors
        indices = jnp.argsort(distance, descending=True)
        d_min = (jnp.sort(distance, descending=True)[: self.max_size])[-1]

        indices = indices[: self.max_size]
        is_offspring_added = jax.vmap(lambda i: jnp.any(indices == i))(
            jnp.arange(self.max_size, self.max_size + batch_of_fitnesses.size)
        )

        # Sort
        genotypes = jax.tree.map(lambda x: x[indices], genotypes)
        descriptors = descriptors[indices]
        passive_descriptors = passive_descriptors[indices]
        fitnesses = fitnesses[indices]
        observations = jax.tree.map(lambda x: x[indices], observations)

        return (
            UnstructuredRepertoirePassiveDesc(
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                passive_descriptors=passive_descriptors,
                observations=observations,
                max_size=self.max_size,
                d_min=d_min,
            ),
            is_offspring_added,
        )

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        passive_descriptors: Descriptor,
        observations: Observation,
        max_size: int,
        d_min: jnp.ndarray,
    ) -> UnstructuredRepertoirePassiveDesc:
        """Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
                genotypes: initial genotypes, pytree in which leaves
                        have shape (batch_size, num_features)
                fitnesses: fitness of the initial genotypes of shape (batch_size,)
                descriptors: descriptors of the initial genotypes
                        of shape (batch_size, num_descriptors)
                passive_descriptors: passive descriptors of the initial genotypes
                        of shape (batch_size, num_passive_descriptors)
                observations: observations experienced in the evaluation task.
                size: size of the repertoire

        Returns:
                an initialized unstructured repertoire.
        """

        # Init repertoire with dummy values
        dummy_genotypes = jax.tree.map(
            lambda x: jnp.full((max_size,) + x.shape[1:], fill_value=jnp.nan),
            genotypes,
        )
        dummy_fitnesses = jnp.full((max_size,), fill_value=-jnp.inf)
        dummy_descriptors = jnp.full(
            (max_size,) + descriptors.shape[1:], fill_value=jnp.nan
        )
        dummy_passive_descriptors = jnp.full(
            (max_size,) + passive_descriptors.shape[1:], fill_value=jnp.nan
        )
        dummy_observations = jax.tree.map(
            lambda x: jnp.full((max_size,) + x.shape[1:], fill_value=jnp.nan),
            observations,
        )

        repertoire = UnstructuredRepertoirePassiveDesc(
            genotypes=dummy_genotypes,
            fitnesses=dummy_fitnesses,
            descriptors=dummy_descriptors,
            passive_descriptors=dummy_passive_descriptors,
            observations=dummy_observations,
            max_size=max_size,
            d_min=d_min,
        )

        repertoire, is_offspring_added = repertoire.add(
            genotypes,
            descriptors,
            passive_descriptors,
            fitnesses,
            observations,
        )
        return repertoire

    def extinction(
        self, remaining_prop: float, random_key
    ) -> UnstructuredRepertoirePassiveDesc:
        """Randomly removes a proportion of individuals from the repertoire.

        Args:
            remaining_prop: proportion of individuals to keep (between 0 and 1)

        Returns:
            A new repertoire with only the remaining individuals
        """

        best_index = jnp.argmax(self.fitnesses)

        # Calculate how many individuals to keep (subtracts 1 for the best individual)
        remaining_count = jnp.maximum(
            1, jnp.floor(self.max_size * remaining_prop).astype(jnp.int32)
        )

        # Generate random mask for valid entries, excluding the best index
        mask = jnp.ones(self.max_size, dtype=bool)
        mask = mask.at[best_index].set(False)  # Remove best_idx from consideration
        available_indices = jnp.where(mask)[0]

        # Shuffle available indices and select remaining_count - 1 of them
        shuffled_available = jax.random.permutation(random_key, available_indices)
        selected_random_indices = shuffled_available[: remaining_count - 1]

        # Combine best_idx with randomly selected indices
        selected_indices = jnp.concatenate(
            [jnp.array([best_index]), selected_random_indices]
        )

        # Create mask for selected entries
        final_mask = jnp.zeros(self.max_size, dtype=bool).at[selected_indices].set(True)

        # Create dummy values for non-selected entries
        dummy_fitnesses = jnp.full_like(self.fitnesses, fill_value=-jnp.inf)
        dummy_descriptors = jnp.full_like(self.descriptors, fill_value=jnp.nan)
        dummy_passive_descriptors = jnp.full_like(
            self.passive_descriptors, fill_value=jnp.nan
        )

        # Update repertoire
        new_fitnesses = jnp.where(final_mask, self.fitnesses, dummy_fitnesses)
        new_descriptors = jnp.where(
            final_mask[..., None], self.descriptors, dummy_descriptors
        )
        new_passive_descriptors = jnp.where(
            final_mask[..., None], self.passive_descriptors, dummy_passive_descriptors
        )

        def mask_genotype(x):
            # Expand mask dimensions to match the genotype shape
            expanded_mask = final_mask
            for _ in range(len(x.shape) - 1):
                expanded_mask = expanded_mask[..., None]
            return jnp.where(expanded_mask, x, jnp.full_like(x, jnp.nan))

        new_genotypes = jax.tree.map(mask_genotype, self.genotypes)
        new_observations = jax.tree.map(mask_genotype, self.observations)

        return UnstructuredRepertoirePassiveDesc(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            passive_descriptors=new_passive_descriptors,
            observations=new_observations,
            max_size=self.max_size,
            d_min=self.d_min,
        )
