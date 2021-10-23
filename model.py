import logging
from pandas import Series, DataFrame
from numpy import array, empty, append

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
                 local_method_parameters: dict = None
                 ):
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

        self.clusters_parameters = None
        self.membership_degrees_current = None

        self._rules_outputs_hist = empty((0, self.clusterization_parameters['n_clusters']), dtype=float)
        self.rules_outputs_hist = DataFrame(dtype=float).copy()
        self.rules_outputs_current = None

        self.current_output = None
        self.hist_output = Series(dtype=float).copy()

        # if test_data is not None:
        #     self.test_data = test_data.copy()
        # else:
        #     if n_test is not None:
        #         self.test_data = self.train_data.iloc[-n_test:].copy()
        #         self.train_data = self.train_data.iloc[:-n_test].copy()
        #     else:
        #         self.test_data = Series(dtype=float).copy()
        #
        # self.logger.debug(f'test_data: {test_data}\n'
        #                   f'train_data: {train_data}')

    def train(self, train_data: Series = None):
        # TODO write train logic
        if train_data is not None:
            self.train_data = train_data.copy()

        # clusterization
        self.logger.debug('Starting clusterization')

        clusterization_result = cluster_data(self.train_data,
                                             method=self.clusterization_method,
                                             parameters=self.clusterization_parameters)

        self.clusters_parameters = clusterization_result['parameters']
        n_clusters = self.clusters_parameters['n_clusters']

        self.membership_degrees_current = clusterization_result['membership']

        self.logger.debug(f'Clusterization completed\n'
                          f'Estimated parameters: {self.clusters_parameters}\n'
                          f'Membership degrees:\n{self.membership_degrees_current}')

        # running local models for each rule
        self.logger.debug('Starting to run local model for each rule')

        self.rules_outputs_current = []
        for i in range(n_clusters):
            rule_output = apply_local_model(train_data,
                                            method=self.local_method,
                                            parameters=self.local_method_parameters,
                                            forecast_horizon=1)['forecast']
            self.rules_outputs_current.append(rule_output)

        rules_outputs_current = array(self.rules_outputs_current)

        self.logger.debug(f'Local model runs for each rule are completed. rules_outputs_current: '
                          f'{self.rules_outputs_current}')

        self._rules_outputs_hist = append(self._rules_outputs_hist, [rules_outputs_current])

        # aggregating rules outputs to a single output
        self.logger.debug('Starting to aggregate all rules outputs to a single one')

        combined_output = combine_rules_outputs(rules_outputs_current, self.membership_degrees_current)

        self.current_output = combined_output

    def push(self, observation):
        # TODO write push logic
        pass

    def forecast(self, test_data):
        # TODO write forecast logic
        pass
