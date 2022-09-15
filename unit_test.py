import unittest
import numpy as np
import jax.numpy as jnp

from sparse_gp import repeat_param_for_kernel, spectral_kernel, get_param_matrices_from_core_params

class TestKernelHelperMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestKernelHelperMethods, self).__init__(*args, **kwargs)
        self.num_comp = 3
        self.Q = 8

    def test_repeat_param_for_kernel(self):
        """
        Test repeat parameter method
        """
        param = jnp.array(np.random.rand(self.num_comp, self.num_comp, self.Q))
        num_x1 = 25; num_x2 = 5
        self.assertEqual(
            repeat_param_for_kernel(param, num_x1, num_x2).shape,
            (75, 15, 8),
            'Produce wrong repetition for kernel params')

        param = jnp.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        num_x1 = 2; num_x2 = 1
        expected_repeat_param = jnp.array([[1, 2], [1, 2], [3, 4], [3, 4]]).reshape(4, 2, 1)
        np.testing.assert_array_equal(
            repeat_param_for_kernel(param, num_x1, num_x2),
            expected_repeat_param,
            'Produce wrong repetition for kernel params')


    def test_spectral_kernel(self):
        """
        Test spectral kernel method generated from "paired" parameter and two set of data points
        """
        X1 = jnp.array(np.random.rand(15)); X2 = jnp.array(np.random.rand(5))
        sigma_ij = jnp.ones((3, 3, 4))
        mu_ij = jnp.ones((3, 3, 4))
        alpha_ij = jnp.ones((3, 3, 4))
        phi_ij = jnp.ones((3, 3, 4))
        theta_ij = jnp.ones((3, 3, 4))
        self.assertEqual(
            spectral_kernel(X1, X2, sigma_ij, mu_ij, alpha_ij, phi_ij, theta_ij).shape,
            (45, 15),
            'Produce wrong spectral kernel')

    def test_get_param_matrices_from_core_params(self):
        pass

if __name__ == '__main__':
    unittest.main()