import unittest
from scipy.stats import multivariate_normal, norm
from numpy.random import randint
from numpy import diag

from scipy_class_inheritance import LongMultivariateNormal


class TestPDF(unittest.TestCase):
    def test_1d_random(self):
        mean = [0]
        cov = [[1]]
        scipy_dist = multivariate_normal(mean=mean, cov=cov)
        custom_dist = LongMultivariateNormal(mean=mean, cov=cov)

        x = norm(0, 1).rvs(1)

        scipy_pdf = scipy_dist.pdf(x)
        custom_pdf = custom_dist.pdf(x)

        round_to = 7

        self.assertAlmostEqual(scipy_pdf, custom_pdf, places=round_to)

    def test_multidim_random(self):
        n = randint(1, 11)

        mean = [0] * n
        cov = diag([1] * n)

        scipy_dist = multivariate_normal(mean=mean, cov=cov)
        custom_dist = LongMultivariateNormal(mean=mean, cov=cov)

        x = norm(0, 1).rvs(1)

        scipy_pdf = scipy_dist.pdf(x)
        custom_pdf = custom_dist.pdf(x)

        round_to = 7

        self.assertAlmostEqual(scipy_pdf, custom_pdf, places=round_to)


if __name__ == '__main__':
    unittest.main()
