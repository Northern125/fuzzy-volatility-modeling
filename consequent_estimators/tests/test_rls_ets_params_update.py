import unittest
from numpy import array, diag

from consequent_estimators.recursive_ls import _ets_new_cluster_re_estimate_parameters


class TestRLSeTSUpdate(unittest.TestCase):
    def test_with_simple_inputs(self):
        weights = array([.2, .8])
        params_prev = array([1, 2, 3, 4, 5, 6])

        cov_prev = diag([1, 2, 3, 4, 5, 6]).copy()
        cov_prev[1, 3] = 10
        cov_prev[3, 1] = 10

        n_params_in_a_rule = 3
        omega = 1e4

        cov_new, params_new = _ets_new_cluster_re_estimate_parameters(params_prev=params_prev,
                                                                      cov_prev=cov_prev,
                                                                      weights=weights,
                                                                      n_params_in_a_rule=n_params_in_a_rule,
                                                                      omega=omega)

        params_new_true = array([1, 2, 3, 4, 5, 6, 3.4, 4.4, 5.4])
        cov_new_true = array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 10, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 10, 0, 4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 6, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, omega, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, omega, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, omega]
        ])

        precision = 1e-5
        self.assertTrue((abs(params_new - params_new_true) < precision).all())
        self.assertTrue((abs(cov_new - cov_new_true) < precision).all().all())


if __name__ == '__main__':
    unittest.main()
