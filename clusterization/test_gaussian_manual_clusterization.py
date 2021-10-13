import unittest
from numpy import array
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


if __name__ == '__main__':
    unittest.main()
