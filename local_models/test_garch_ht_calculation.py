import unittest
from scipy.stats import norm
from numpy import array
import logging

from . import calc_cond_var, calc_ht, calc_fuzzy_ht

logging.basicConfig(level=logging.DEBUG)


class TestCustomGARCHCondVarCalc(unittest.TestCase):
    def test_vanilla_garch_1_1(self):
        logger = logging.getLogger('test_vanilla_garch_1_1')

        # generating white noise
        sample_size = 1
        mean = 0
        variance = 1
        data = norm(mean, variance ** .5).rvs(sample_size)

        # garch parameters
        alpha_0 = 1
        alpha = array([2])
        beta = array([4])
        y_squared = data ** 2
        first_h = array([1])

        # calculating
        h_main_fun = calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                                   fuzzy=False, weights=None)
        h_aux_fun = calc_ht(alpha_0, alpha, beta, y_squared, first_h)
        logger.debug(f'alpha_0 = {alpha_0}, alpha = {alpha}, beta = {beta}, y_squared = {y_squared}, '
                     f'first_h = {first_h}')
        logger.debug(f'h_main_fun = {h_main_fun}')

        tolerance = 1e-5
        self.assertAlmostEqual(h_main_fun[1], h_aux_fun, delta=tolerance)

    def test_fuzzy_garch_1_1(self):
        logger = logging.getLogger('test_fuzzy_garch_1_1')

        # generating white noise
        sample_size = 1
        mean = 0
        variance = 1
        data = norm(mean, variance ** .5).rvs(sample_size)

        # garch parameters
        alpha_0 = array([1])
        alpha = array([[2]])
        beta = array([[4]])
        y_squared = data ** 2
        first_h = array([1])

        weights = array([1])

        # calculating
        h_main_fun = calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                                   fuzzy=True, weights=weights)
        h_aux_fun = calc_fuzzy_ht(alpha_0, alpha, beta, y_squared, first_h, weights)
        logger.debug(f'alpha_0 = {alpha_0}, alpha = {alpha}, beta = {beta}, y_squared = {y_squared}, '
                     f'first_h = {first_h}')
        logger.debug(f'h_main_fun = {h_main_fun}')

        tolerance = 1e-5
        self.assertAlmostEqual(h_main_fun[1], h_aux_fun, delta=tolerance)


if __name__ == '__main__':
    unittest.main()
