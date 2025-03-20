import jax
import jax.numpy as jnp


def mse_loss_fn(logits: jnp.ndarray, batch: jnp.ndarray) -> jnp.ndarray:
    def mean_squared_error(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.inner(y - x, y - x) / x.shape[-1]

    res = jax.vmap(mean_squared_error)(
        jnp.reshape(logits.at[:, :-1, ...].get(), (logits.shape[0], -1)),
        jnp.reshape(batch.at[:, 1:, ...].get(), (batch.shape[0], -1)),
    )
    return jnp.mean(res, axis=0)


def triplet_loss_fn(
    anchor: jnp.ndarray, pos: jnp.ndarray, neg: jnp.ndarray, alpha: float
) -> jnp.ndarray:

    def squared_error(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.inner(y - x, y - x)

    loss = jnp.maximum(
        0, squared_error(anchor, pos) ** 2 - squared_error(anchor, neg) ** 2 + alpha
    )
    return loss
