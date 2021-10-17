import unittest
from scipy.stats import multivariate_normal, norm

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


if __name__ == '__main__':
    unittest.main()
