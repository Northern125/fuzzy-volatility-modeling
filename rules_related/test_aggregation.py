import unittest
from scipy.stats import norm
from numpy.random import randint
from numpy import array

from aggregation import combine_rules_outputs


class TestAggregation(unittest.TestCase):
    def test_random_number_equal_weights(self):
        random_number = norm(0, 100).rvs(1)[0]
        random_n = randint(1, 101)

        outputs = array([random_number] * random_n)
        weights = array([1] * random_n)

        result = combine_rules_outputs(outputs, weights)

        round_to = 5

        self.assertEqual(round(result, round_to), round(random_number, round_to),
                         f'result = {result}, random_number = {random_number}')


if __name__ == '__main__':
    unittest.main()
