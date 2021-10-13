import unittest
from numpy import array, exp
from math import pi

from gaussian import calc_gaussian_membership_degrees


class TestGaussianMembershipFunction(unittest.TestCase):
    def test_1d_1_cluster(self):
        input_data = array([0])
        centers = array([[0]])
        cov_matrices = array([[[1]]])

        result = calc_gaussian_membership_degrees(input_data, centers, cov_matrices)[0]

        should_be = 1 / ((2 * pi) ** .5)

        self.assertEqual(result, should_be)

    def test_1d_3_clusters(self):
        input_data = array([0])
        centers = array([[-1], [0], [1]])
        cov_matrices = array([[[1]], [[1]], [[1]]])

        result = calc_gaussian_membership_degrees(input_data, centers, cov_matrices)

        coeff = 1 / ((2 * pi) ** .5)
        e_to_one_half = exp(-.5)
        should_be = [coeff * e_to_one_half, coeff, coeff * e_to_one_half]

        assertion = (result == should_be).all()

        self.assertTrue(assertion)


if __name__ == '__main__':
    unittest.main()
