import logging
from pandas import Series, DataFrame
from numpy import array

from clusterization import cluster_data
from local_models import apply_local_model
from rules_related import combine_rules_outputs

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

    def fit(self, train_data: Series = None):
        if train_data is not None:
            self.train_data = train_data.copy()

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

        # running local models for each rule
        self.logger.debug('Starting to run local model for each rule')

        self.rules_outputs_current = []
        for i in range(n_clusters):
            rule_output = apply_local_model(self.train_data,
                                            method=self.local_method,
                                            parameters=self.local_method_parameters,
                                            forecast_horizon=1)['forecast'][0]  # yes, hardcode:
            # we only forecast for the next day, it is a common practice
            self.rules_outputs_current.append(rule_output)

        self.rules_outputs_current = array(self.rules_outputs_current)

        self.logger.debug(f'Local model runs for each rule are completed. rules_outputs_current: '
                          f'{self.rules_outputs_current}')

        self._rules_outputs_hist.append(self.rules_outputs_current)

        # aggregating rules outputs to a single output
        self.logger.debug('Starting to aggregate all rules outputs to a single one')

        self.current_output = combine_rules_outputs(self.rules_outputs_current, self.membership_degrees_current)
        self.logger.debug(f'Rules outputs are combined; current_output: {self.current_output}')
        self._hist_output.append(self.current_output)

    def push(self, observation: float, observation_date):
        self.train_data.loc[observation_date] = observation
        self.fit()

    def forecast(self, test_data: Series):
        for date in test_data.index:
            observation = test_data.loc[date]
            self.push(observation, date)
