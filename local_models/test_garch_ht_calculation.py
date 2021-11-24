import unittest
from scipy.stats import norm
from numpy import array

from . import calc_cond_var, calc_ht


class TestCustomGARCHCondVarCalc(unittest.TestCase):
    def test_vanilla_garch_1_1(self):
        # generating white noise
        sample_size = 100
        mean = 0
        variance = 1
        data = norm(mean, variance ** .5).rvs(sample_size)

        n = data.shape[0]

        # garch parameters
        p = 1
        q = 1
        alpha_0 = 1
        alpha = array([2])
        beta = array([4])
        y_squared = array([6])
        first_h = array([1])

        h_main_fun = calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                                   fuzzy=False, weights=None)
        print(h_main_fun)
        h_aux_fun = calc_ht(alpha_0, alpha, beta,
                            y_squared, first_h)
        print(h_aux_fun)

        self.assertEqual(h_main_fun[0], h_aux_fun)


if __name__ == '__main__':
    unittest.main()
