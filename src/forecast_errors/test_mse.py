import unittest
from sklearn.metrics import mean_squared_error


class TestSKLearnMSE(unittest.TestCase):
    def test_mse_rmse(self):
        estimates = [1, 2, 3]
        true = [2, -1, 5]
        mse = 14 / 3
        rmse = mse ** .5

        mse_sklearn = mean_squared_error(true, estimates, squared=True)
        rmse_sklearn = mean_squared_error(true, estimates, squared=False)

        tolerance = 1e-5
        self.assertAlmostEqual(mse, mse_sklearn, delta=tolerance)
        self.assertAlmostEqual(rmse, rmse_sklearn, delta=tolerance)


if __name__ == '__main__':
    unittest.main()
