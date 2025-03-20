import jax.numpy as jnp
import jax
from omegaconf import DictConfig
import hydra
import flax
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.neuroevolution.networks.networks import MLP
from jax.flatten_util import ravel_pytree

from ae_utils.model_train import init_autoencoder_model_training
from typing import Callable, List, Optional, Tuple, Union
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
    Observation,
)


class Repertoire(MapElitesRepertoire):
    genotypes: Genotype
    fitnesses: Fitness
    descriptors: Descriptor
    centroids: Centroid
    observations: Observation
    max_size: int = flax.struct.field(pytree_node=False)

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> MapElitesRepertoire:
        """Loads a MAP Elites Repertoire.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)

        fitnesses = jnp.load(path + "fitnesses.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        centroids = jnp.load(path + "centroids.npy")
        observations = jnp.load(path + "observations.npy")
        max_size = fitnesses.shape[0]

        return cls(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            observations=observations,
            max_size=max_size,
        )


def train(cfg: DictConfig, repertoire: MapElitesRepertoire, random_key: jnp.ndarray):
    observations_dims = (64, 64, 3)
    random_key, subkey = jax.random.split(random_key)
    cfg.env.observation_extraction.observation_option = "images"
    cfg.learning_rate = cfg.lr
    cfg.model_epoch_period = cfg.epochs
    cfg.model_batch_size = cfg.batch_size
    encoder_fn, train_fn, train_state, aurora_extra_info, model = (
        init_autoencoder_model_training(cfg, observations_dims, random_key)
    )
    train_state, aurora_extra_info, _ = train_fn(
        repertoire, train_state, random_key=subkey, iteration=0
    )


@hydra.main(config_path="../configs", config_name="ae_train.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    path_dir = "dataset/"
    policy_layer_sizes = (5, 2)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of policies
    random_key = jax.random.PRNGKey(cfg.seed)
    random_key, subkey = jax.random.split(random_key)
    fake_batch = jnp.zeros(shape=(5,))
    fake_params = policy_network.init(subkey, fake_batch)

    # Load the repertoire
    _, reconstruction_fn = ravel_pytree(fake_params)
    repertoire = Repertoire.load(
        reconstruction_fn=reconstruction_fn, path=path_dir
    )
    train(repertoire=repertoire, cfg=cfg, random_key=random_key)


if __name__ == "__main__":
    main()
