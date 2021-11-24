import unittest
from scipy.stats import norm
from arch import arch_model


class TestCustomGARCH(unittest.TestCase):
    def test_vanilla_garch_vs_arch_package(self):
        # generating white noise
        sample_size = 100
        mean = 0
        variance = 1
        data = norm(mean, variance ** .5).rvs(sample_size)

        n = data.shape[0]

        # garch parameters
        p = 1
        q = 1

        # package model
        model = arch_model(data,
                           mean='Zero',
                           vol='GARCH',
                           p=p,
                           q=q,
                           dist='normal')

        # my custom model


if __name__ == '__main__':
    unittest.main()
