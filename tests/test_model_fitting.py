import unittest

from arch import arch_model
from scipy.stats import norm
from pandas import date_range, Series, Timestamp
from numpy import random, array, float128

from model import FuzzyVolatilityModel


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

centers = [mu]
variances = [sigma ** 2]

clusterization_parameters = {'centers': centers, 'variances': variances, 'n_clusters': 1}


class TestFit(unittest.TestCase):
    def test_1_cluster_fitting_via_1_day_forecast(self):
        # creating, fitting and running our model
        fvm = FuzzyVolatilityModel(data,
                                   clusterization_method=clusterization_method,
                                   clusterization_parameters=clusterization_parameters,
                                   local_method=local_method,
                                   local_method_parameters=local_method_parameters)
        fvm.fit()
        fvm.forecast()
        fvm_forecasted_variance = fvm.current_output
        fvm_rules_outputs = fvm.rules_outputs_current

        tolerance = 1e-5
        self.assertAlmostEqual(fvm_forecasted_variance, fvm_rules_outputs[0], delta=tolerance)

        # creating, fitting and running garch
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

    def test_1_cluster_feeding(self):
        n_test = random.randint(1, high=11)
        n_train = n - n_test
        train, test = data.iloc[:n_train].copy(), data.iloc[n_train:].copy()

        # creating, fitting and running our model
        fvm = FuzzyVolatilityModel(train,
                                   clusterization_method=clusterization_method,
                                   clusterization_parameters=clusterization_parameters,
                                   local_method=local_method,
                                   local_method_parameters=local_method_parameters)
        fvm.fit()
        fvm.forecast()
        fvm.feed_daily_data(test)
        single_rule_output = array(fvm._rules_outputs_hist, dtype=float128)[:, 0]
        total_output = array(fvm._hist_output, dtype=float128)

        tolerance = 5
        self.assertTrue((single_rule_output.round(tolerance) == total_output.round(tolerance)).all())


if __name__ == '__main__':
    unittest.main()
