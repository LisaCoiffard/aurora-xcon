import jax
import jax.numpy as jnp
import unittest
from new_ae import AutoEncoder


class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        self.model = AutoEncoder(img_shape=(64, 64, 3), latent_size=10, features=128)
        self.params = self.model.init(
            jax.random.PRNGKey(0), jnp.ones((64, 64, 3)), jax.random.PRNGKey(1)
        )
        # print the shape of each kernel in the model
        for k in self.params["params"].keys():
            for k_i in self.params["params"][k].keys():
                print(k_i, self.params["params"][k][k_i]["kernel"].shape)

    def test_autoencoder(self):
        x = jnp.ones((16, 64, 64, 3))

        enc = self.model.apply(
            self.params, x, jax.random.PRNGKey(2), method=self.model.encode
        )
        self.assertEqual(enc.shape, (16, 10))

        y = self.model.apply(self.params, x, jax.random.PRNGKey(3))

        # Check the output shape
        self.assertEqual(y.shape, (16, 64, 64, 3))

        # Check that the output is not all ones
        self.assertFalse(jnp.all(y == 1))



if __name__ == "__main__":
    unittest.main()
