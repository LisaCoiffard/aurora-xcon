from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit

import qdax_es.core.containers.gp_repertoire
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
)
from qdax.utils.plotting import plot_2d_map_elites_repertoire
from qdax_es.core.containers.count_repertoire import CountMapElitesRepertoire
from qdax_es.utils.count_plots import plot_2d_count, plot_archive_value
from qdax_es.utils.gaussian_process import (
    GPState,
    gp_batch_predict,
    gp_predict,
    train_gp,
)


class GPRepertoire(CountMapElitesRepertoire):
    gp_state: GPState = None
    
    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        extra_scores: Optional[ExtraScores] = None,
        weighted: bool = False,
        max_count: int = 1e4,
    ) -> CountMapElitesRepertoire:
        """Initialize a repertoire"""

        # retrieve one genotype from the population
        first_genotype = jax.tree_util.tree_map(lambda x: x[0], genotypes)

        # create a repertoire with default values
        repertoire = cls.init_default(genotype=first_genotype, centroids=centroids)

        # add initial population to the repertoire
        repertoire, condition = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        gp_state = GPState.init_from_repertoire(repertoire, weighted)
        return repertoire.replace(gp_state=gp_state), condition

    @jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> CountMapElitesRepertoire:
        
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

        num_centroids = self.centroids.shape[0]

        count = self.count + jnp.bincount(
            batch_of_indices.squeeze(axis=-1),
            minlength=len(self.count),
            length=len(self.count),
        )
        # count = self.count + self._count(batch_of_indices)

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
        )

        # get addition condition
        repertoire_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(
            repertoire_fitnesses, batch_of_indices, 0
        )
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, batch_of_indices, num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree_util.tree_map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses.squeeze(axis=-1)
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )

        new_repertoire = self.__class__(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            count=count,
            gp_state=self.gp_state,
        )

        return new_repertoire, addition_condition
    
    @jit
    def __add__(
        self,
        other_repertoire: CountMapElitesRepertoire,
    ) -> CountMapElitesRepertoire:
        # Super add
        new_repertoire = super().__add__(other_repertoire)
        # set ls_scaler
        return new_repertoire.replace(
            gp_state=self.gp_state,
        )

    @partial(jit, static_argnames=("n_steps",))
    def fit_gp(self, n_steps: int = 1000):
        gp_state = GPState.init_from_repertoire(self, self.gp_state.weighted)
        fit_gp_state = train_gp(gp_state, num_steps=n_steps)
        return self.replace(gp_state=fit_gp_state)

    @jit
    def predict(self, x_new):
        return gp_predict(self.gp_state, x_new)
    
    @jit
    def batch_predict(self, x_new):
        return gp_batch_predict(self.gp_state, x_new)


    def plot(
            self,
            min_bd,
            max_bd,
            title='GP',
            plot_gp=True,
            cfg=None,
            ):
        """Plot the repertoire"""
        if plot_gp:
            fig, axes = plt.subplot_mosaic("""
                    AB
                    CD
                    """,
                    figsize=(20, 15),
                )
        else:
            fig, axes = plt.subplot_mosaic("""
                AB
                """,
                figsize=(20, 8),
            )
        try:
            vmin, vmax = None, None
            if cfg is not None:
                vmin, vmax = cfg.task.plotting.fitness_bounds
            _, axes["A"] = plot_2d_map_elites_repertoire(
                centroids=self.centroids,
                repertoire_fitnesses=self.fitnesses,
                minval=min_bd,
                maxval=max_bd,
                repertoire_descriptors=self.descriptors,
                ax=axes["A"],
                vmin=vmin,
                vmax=vmax,
            )
            max_fit = jnp.max(self.fitnesses)
            axes["A"].set_title(f"Fitness (max: {max_fit:.2f})")

            vmin, vmax = None, None
            if cfg is not None:
                vmin, vmax = 0, cfg.task.plotting.max_eval_cell
            axes["B"] = plot_2d_count(
                self, 
                min_bd, 
                max_bd, 
                log_scale=True, 
                ax=axes["B"],
                colormap="plasma",
                vmin=vmin,
                vmax=vmax,
                )
            
            if plot_gp:
                # print(f"Plot GP LS: {self.gp_params.lengthscale}")
                means, covs = self.batch_predict(self.centroids)

                _, axes["C"] = plot_archive_value(
                    self, 
                    means, 
                    min_bd, 
                    max_bd,
                    ax=axes["C"],
                    title="GP mean"
                )
                _, axes["D"] = plot_archive_value(
                    self, 
                    covs, 
                    min_bd, 
                    max_bd,
                    ax=axes["D"],
                    title="GP variance"
                )
            plt.suptitle(title, fontsize=20)
        except Exception as e:
            # raise e
            print("Failed plotting")

        return fig, axes
    
    def plot_gp(
            self,
            min_bd,
            max_bd,
    ):
        """Plot only GP as 2 separate plots"""
    
        means, covs = self.batch_predict(self.centroids)
        # Plot GP mean
        mean_fig, mean_ax = plt.subplots(figsize=(10, 10))
        mean_ax = plot_archive_value(
            self, 
            means, 
            min_bd, 
            max_bd,
            ax=mean_ax,
            title="GP mean"
        )

        # Plot GP variance
        var_fig, var_ax = plt.subplots(figsize=(10, 10))
        var_ax = plot_archive_value(
            self, 
            covs, 
            min_bd, 
            max_bd,
            ax=var_ax,
            title="GP variance"
        )

        return mean_fig, var_fig