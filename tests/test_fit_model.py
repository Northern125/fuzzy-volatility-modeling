import unittest

from arch import arch_model
from scipy.stats import norm
from pandas import date_range, Series, Timestamp

from model import FuzzyVolatilityModel


class TestFit(unittest.TestCase):
    def test_1_cluster_fitting(self):
        # generating white noise
        sample_size = 100
        mean = 0
        variance = 1
        data = norm(mean, variance ** .5).rvs(sample_size)

        n = data.shape[0]
        dates = date_range(start=Timestamp.now().floor('D'), periods=n)

        data = Series(data, index=dates).copy()

        # setting parameters for our model
        local_method = 'garch'
        local_method_parameters = {'p': 1, 'q': 1, 'mean': 'Zero', 'dist': 'normal'}

        clusterization_method = 'gaussian'

        mu = mean
        sigma = variance ** .5

        centers = [mu]  # np.array([[mu] * n])
        variances = [sigma ** 2]  # np.array([np.diag([sigma] * n, k=0)])

        clusterization_parameters = {'centers': centers, 'variances': variances, 'n_clusters': 1}

        # running our model
        fvm = FuzzyVolatilityModel(data,
                                   clusterization_method=clusterization_method,
                                   clusterization_parameters=clusterization_parameters,
                                   local_method=local_method,
                                   local_method_parameters=local_method_parameters)
        fvm.fit()
        fvm_forecasted_variance = fvm.current_output

        # running garch
        garch_model = arch_model(data,
                                 mean=local_method_parameters['mean'],
                                 vol='GARCH', p=local_method_parameters['p'],
                                 q=local_method_parameters['q'],
                                 dist=local_method_parameters['dist'])
        fitted = garch_model.fit()
        garch_forecast = fitted.forecast(reindex=False, horizon=1)
        garch_forecasted_variance = garch_forecast.variance.values[0][0]

        round_to = 5

        self.assertAlmostEqual(fvm_forecasted_variance, garch_forecasted_variance, places=round_to)


if __name__ == '__main__':
    unittest.main()
