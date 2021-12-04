import logging

from arch import arch_model
from pandas import Series, DataFrame, concat
from numpy import array
from scipy.optimize import least_squares

from clusterization import cluster_data
from rules_related import combine_rules_outputs
from local_models import calc_cond_var
from auxiliary import unpack_1d_parameters, pack_1d_parameters

module_logger = logging.getLogger('model')


class FuzzyVolatilityModel:
    def __init__(self,
                 train_data: Series = None,
                 clusterization_method: str = 'gaussian',
                 clusterization_parameters: dict = None,
                 local_method: str = 'garch',
                 local_method_parameters: dict = None):
        self.logger = logging.getLogger(module_logger.name + '.FuzzyVolatilityModel')
        self.logger.info('Creating an instance of FuzzyVolatilityModel')

        self.clusterization_method = clusterization_method
        self.clusterization_parameters = clusterization_parameters
        self.local_method = local_method
        self.local_method_parameters = local_method_parameters

        if train_data is None:
            self.train_data = Series(dtype=float).copy()
        else:
            self.train_data = train_data.copy()

        # clusters parameters
        self._clusters_parameters_hist = []
        self.clusters_parameters_hist = DataFrame(dtype=float).copy()
        self.clusters_parameters_current = None

        # membership degrees
        self._membership_degrees_hist = []
        self.membership_degrees_hist = DataFrame(dtype=float).copy()
        self.membership_degrees_current = None

        # rules outputs
        self._rules_outputs_hist = []
        self.rules_outputs_hist = DataFrame(dtype=float).copy()
        self.rules_outputs_current = None

        # combined output
        self._hist_output = []
        self.hist_output = Series(dtype=float).copy()
        self.current_output = None

        # garch objects
        # self.garch_models = []
        # self.fitted_garch_models = []

        # consequent parameters
        self.alpha_0 = None
        self.alpha = None
        self.beta = None
        self._alpha_0_hist = None
        self._alpha_hist = None
        self._beta_hist = None
        self.alpha_0_hist = None
        self.alpha_hist = None
        self.beta_hist = None

    def fit(self, train_data: Series = None):
        if train_data is not None:
            self.train_data = train_data.copy()

        n = self.train_data.shape[0]

        # clusterization
        self.logger.debug('Starting clusterization')

        clusterization_result = cluster_data(self.train_data,
                                             method=self.clusterization_method,
                                             parameters=self.clusterization_parameters)

        self.clusters_parameters_current = clusterization_result['parameters']
        n_clusters = self.clusters_parameters_current['n_clusters']

        self.membership_degrees_current = clusterization_result['membership']

        self.logger.debug(f'Clusterization completed\n'
                          f'Estimated parameters: {self.clusters_parameters_current}\n'
                          f'Membership degrees:\n{self.membership_degrees_current}')

        self._clusters_parameters_hist.append(self.clusters_parameters_current)
        self._membership_degrees_hist.append(self.membership_degrees_current)

        # fitting local models within each rule
        self.logger.debug('Starting to fit local model within each rule')

        p = self.local_method_parameters['p']
        q = self.local_method_parameters['q']
        mean = self.local_method_parameters['mean']
        dist = self.local_method_parameters['dist']
        first_h = array(self.local_method_parameters['first_h'])
        bounds = self.local_method_parameters['bounds']
        parameters_ini = self.local_method_parameters['parameters_ini']

        starting_index = max(p, q)
        self.logger.debug(f'starting_index = {starting_index}')

        def calc_residuals(_parameters):
            alpha_0, alpha, beta = unpack_1d_parameters(_parameters, p=p, q=q, n_clusters=n_clusters)

            h = calc_cond_var(alpha_0, alpha, beta, self.train_data ** 2, first_h,
                              fuzzy=True, weights=self.membership_degrees_current)

            residuals = self.train_data[starting_index:] - h[starting_index:-1]
            self.logger.debug(f'residuals =\n{residuals}')
            self.logger.debug(f'RSS = {(residuals ** 2).sum()}')
            return residuals

        alpha_0_ini, alpha_ini, beta_ini = parameters_ini['alpha_0'], parameters_ini['alpha'], parameters_ini['beta']
        parameters_0 = pack_1d_parameters(alpha_0_ini, alpha_ini, beta_ini)
        ls_result = least_squares(calc_residuals, parameters_0, bounds=bounds)

        parameters = ls_result.x
        self.logger.debug(f'parameters = {parameters}')

        self.alpha_0, self.alpha, self.beta = unpack_1d_parameters(parameters, p=p, q=q, n_clusters=n_clusters)

        self.logger.debug('Fitting is completed')

    def _calc_local_models_forecasts(self, horizon=1):
        rules_outputs = []

        for model in self.fitted_garch_models:
            rule_output = model.forecast(reindex=False, horizon=horizon).variance.iloc[0]
            rules_outputs.append(rule_output)

        return rules_outputs

    def forecast(self):
        # calculating rules outputs
        self.logger.debug('Starting to calculate rules outputs')

        rules_outputs = self._calc_local_models_forecasts(horizon=1)
        self.rules_outputs_current = [output.loc['h.1'] for output in rules_outputs]
        self.logger.debug(f'Rules outputs calculated; rules_outputs_current: {self.rules_outputs_current}')
        self._rules_outputs_hist.append(self.rules_outputs_current)

        # aggregating rules outputs to a single output
        self.logger.debug('Starting to aggregate all rules outputs to a single one')

        self.current_output = combine_rules_outputs(self.rules_outputs_current, self.membership_degrees_current)
        self.logger.debug(f'Rules outputs are combined; current_output: {self.current_output}')
        self._hist_output.append(self.current_output)

    def _push(self, observation: float, observation_date):
        self.train_data.loc[observation_date] = observation
        self.fit()

    def feed_daily_data(self, test_data: Series):
        if self.current_output is None:
            # if there is no current forecast (AKA the model has just been created),
            # then do forecast before running the main algorithm
            self.forecast()

        # imitating live daily algorithm work
        for date in test_data.index:
            observation = test_data.loc[date]
            self._push(observation, date)
            self.forecast()

        # adding dates
        dates = test_data.index.copy()
        n_test_dates = dates.shape[0]

        slc = slice(-n_test_dates - 1, -1)

        hist_output_new = Series(self._hist_output[slc], index=dates).copy()
        self.hist_output = concat([self.hist_output, hist_output_new]).copy()

        rules_outputs_hist_new = DataFrame.from_records(self._rules_outputs_hist[slc], index=dates).copy()
        self.rules_outputs_hist = concat([self.rules_outputs_hist, rules_outputs_hist_new], axis='index').copy()

        membership_degrees_hist_new = DataFrame.from_records(self._membership_degrees_hist[slc], index=dates).copy()
        self.membership_degrees_hist = concat([self.membership_degrees_hist, membership_degrees_hist_new],
                                              axis='index').copy()

        clusters_parameters_hist_new = DataFrame.from_records(self._clusters_parameters_hist[slc], index=dates).copy()
        self.clusters_parameters_hist = concat([self.clusters_parameters_hist, clusters_parameters_hist_new],
                                               axis='index').copy()
