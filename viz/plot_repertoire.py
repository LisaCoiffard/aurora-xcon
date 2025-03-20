from typing import Optional, Tuple

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_2d_unstructured_repertoire(
    repertoire_fitnesses: jnp.ndarray,
    repertoire_descriptors: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Axes]:

    grid_empty = repertoire_fitnesses == -jnp.inf

    my_cmap = cm.viridis

    # set the parameters
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "figure.figsize": [10, 10],
    }

    mpl.rcParams.update(params)

    # create the plot object
    fig = None
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")
    ax.set(adjustable="box", aspect="equal")

    # aesthetic
    divider = make_axes_locatable(ax)
    ax.set_aspect("equal")

    # if the grid is empty, plot an empty grid
    if jnp.all(grid_empty):
        return fig, ax

    fitnesses = repertoire_fitnesses
    if vmin is None:
        vmin = float(jnp.min(fitnesses[~grid_empty]))
    if vmax is None:
        vmax = float(jnp.max(fitnesses[~grid_empty]))

    norm = Normalize(vmin=vmin, vmax=vmax)

    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
    cbar.ax.tick_params(labelsize=font_size)

    descriptors = repertoire_descriptors[~grid_empty]
    ax.scatter(
        descriptors[:, 0],
        descriptors[:, 1],
        c=fitnesses[~grid_empty],
        cmap=my_cmap,
        s=10,
        zorder=0,
    )

    return fig, ax


def plot_repertoire_embeddings(
    repertoire_fitnesses: jnp.ndarray,
    repertoire_descriptors: jnp.ndarray,
    embeddings_2d: jnp.ndarray,
    minval: float,
    maxval: float,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Axes]:
    my_cmap = "viridis"

    # Plot the embeddings
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[1].set_xlim(minval, maxval)
    axs[1].set_ylim(minval, maxval)
    axs[0].set_title("AURORA rep. (TSNE plot)")
    axs[1].set_title("Corresponding passive descriptors")

    norm = Normalize(vmin=vmin, vmax=vmax)
    # colours = jnp.arctan2(embeddings_2d[:, 1], embeddings_2d[:, 0])
    colours = norm(repertoire_fitnesses)
    axs[0].scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colours,
        cmap=my_cmap,
        alpha=0.7,
        s=10,
    )

    # Plot the passive repertoire as a scatter plot with same color scheme
    descriptors = repertoire_descriptors
    axs[1].scatter(
        descriptors[:, 0],
        descriptors[:, 1],
        c=colours,
        cmap=my_cmap,
        alpha=0.7,
        s=10,
    )
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), ax=axs)
    return fig, axs
