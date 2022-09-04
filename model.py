import logging
from typing import Union

from pandas import Series, DataFrame, concat
from numpy import array
from scipy.optimize import least_squares

from clusterization import cluster_data
from local_models import calc_cond_var_fuzzy
from auxiliary import unpack_1d_parameters, pack_1d_parameters

module_logger = logging.getLogger(__name__)


class FuzzyVolatilityModel:
    def __init__(self,
                 train_data: Series = None,
                 clusterization_method: str = 'gaussian',
                 clusterization_parameters: dict = None,
                 local_method: str = 'garch',
                 local_method_parameters: dict = None,
                 data_to_cluster: Union[str, Series, DataFrame] = 'train',
                 n_last_points_to_use_for_clustering: int = None,
                 cluster_sets_conjunction: Union['str', callable] = 'prod',
                 n_cluster_sets: int = None,
                 normalize: bool = False):
        self.logger = logging.getLogger(module_logger.name + '.' + type(self).__name__)
        self.logger.info(f'Creating an instance of {self.logger.name}')

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
        self.n_clusters = None

        if type(data_to_cluster) is not str and data_to_cluster is not None:
            self.data_to_cluster = data_to_cluster.copy()
            if type(self.data_to_cluster) is list:
                self.data_to_cluster = [self.train_data.copy()
                                        if type(_elem) is str and _elem == 'train' or _elem is None
                                        else _elem
                                        for _elem in self.data_to_cluster]
                self.data_to_cluster = concat(self.data_to_cluster, axis='columns').copy()
        elif data_to_cluster == 'train' or data_to_cluster is None:
            self.data_to_cluster = self.train_data.copy()
        else:
            raise ValueError("""`data_to_cluster` should be either a string 'train' or not a string""")

        self.n_last_points_to_use_for_clustering = n_last_points_to_use_for_clustering
        self.cluster_sets_conjunction = cluster_sets_conjunction
        self.n_cluster_sets = n_cluster_sets
        self.normalize = normalize

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
        self._parameters_hist = []

        # GARCH variables
        self.h = None

    @staticmethod
    def _convert_data_to_cluster(data_to_cluster, train_data, variable_name=None):
        if type(data_to_cluster) is not str and data_to_cluster is not None:
            converted = data_to_cluster.copy()
            if type(converted) is list:
                converted = [train_data.copy()
                             if type(_elem) is str and _elem == 'train' or _elem is None
                             else _elem
                             for _elem in converted]
                converted = concat(converted, axis='columns').copy()
        elif data_to_cluster == 'train' or data_to_cluster is None:
            # for backward compatibility
            converted = train_data.copy()
        else:
            raise ValueError(f"""`{variable_name if variable_name is not None else 'data'}` 
                                 should be either a string 'train' or not a string""")

        return converted

    def _calc_residuals(self, _parameters, p, q, first_h, starting_index):
        alpha_0, alpha, beta = unpack_1d_parameters(_parameters, p=p, q=q, n_clusters=self.n_clusters)

        h = calc_cond_var_fuzzy(alpha_0, alpha, beta, self.train_data ** 2, first_h,
                                weights=self.membership_degrees_current)

        residuals = self.train_data[starting_index:] ** 2 - h[starting_index:-1]
        self.logger.debug(f'residuals =\n{residuals}')
        self.logger.debug(f'RSS = {(residuals ** 2).sum()}')

        return residuals

    def fit(self, train_data: Series = None):
        if train_data is not None:
            self.train_data = train_data.copy()

        # clusterization
        self.logger.debug('Starting clusterization')

        clusterization_result = cluster_data(self.data_to_cluster,
                                             methods=self.clusterization_method,
                                             parameters=self.clusterization_parameters,
                                             n_last_points_to_use_for_clustering=
                                             self.n_last_points_to_use_for_clustering,
                                             conjunction=self.cluster_sets_conjunction,
                                             n_sets=self.n_cluster_sets,
                                             normalize=self.normalize)

        self.clusters_parameters_current = clusterization_result['parameters']
        self.n_clusters = self.clusters_parameters_current['n_clusters']

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
        first_h = array(self.local_method_parameters['first_h'])
        bounds = self.local_method_parameters['bounds']
        parameters_ini = self.local_method_parameters['parameters_ini']

        starting_index = max(p, q)
        self.logger.debug(f'starting_index = {starting_index}')

        alpha_0_ini, alpha_ini, beta_ini = parameters_ini['alpha_0'], parameters_ini['alpha'], parameters_ini['beta']
        parameters_0 = pack_1d_parameters(alpha_0_ini, alpha_ini, beta_ini)
        self.logger.debug(f'Starting least squares estimation of parameters; `parameters_0`: {parameters_0}')
        ls_result = least_squares(self._calc_residuals, parameters_0, bounds=bounds,
                                  args=(p, q, first_h, starting_index))

        parameters = ls_result.x
        self.logger.debug(f'Least squares estimation finished; estimated parameters = {parameters}')

        self.alpha_0, self.alpha, self.beta = unpack_1d_parameters(parameters, p=p, q=q, n_clusters=self.n_clusters)
        self._parameters_hist.append({'alpha_0': self.alpha_0, 'alpha': self.alpha, 'beta': self.beta})

        self.logger.debug('Fitting is completed')

    def forecast(self):
        first_h = array(self.local_method_parameters['first_h'])
        self.h = calc_cond_var_fuzzy(self.alpha_0, self.alpha, self.beta, self.train_data ** 2, first_h,
                                     weights=self.membership_degrees_current)
        self.current_output = self.h[-1]
        self._hist_output.append(self.current_output)

    def _push(self, observation: float, observation_date, data_to_cluster_point):
        self.train_data.loc[observation_date] = observation
        if type(self.data_to_cluster) is not str:
            self.data_to_cluster.loc[observation_date] = data_to_cluster_point
        elif self.data_to_cluster != 'train':
            raise ValueError("""`data_to_cluster` should be either a string 'train' or not a string""")
        self.fit()

    def feed_daily_data(self, test_data: Series, data_to_cluster=None):
        if self.current_output is None:
            # if there is no current forecast (AKA the model has just been created),
            # then do forecast before running the main algorithm
            self.forecast()

        data_to_cluster = self._convert_data_to_cluster(data_to_cluster,
                                                        test_data,
                                                        variable_name='self.data_to_cluster')
        if (len(data_to_cluster.shape) != len(self.data_to_cluster.shape)) or \
                (len(data_to_cluster.shape) == len(self.data_to_cluster.shape) == 2 and
                 data_to_cluster.shape[1] != self.data_to_cluster.shape[1]):
            raise ValueError('# of columns is different in given `data_to_cluster` and `self.data_to_cluster`')

        # imitating live daily algorithm work
        for date in test_data.index:
            observation = test_data.loc[date]
            data_to_cluster_point = data_to_cluster.loc[date]
            self._push(observation, date, data_to_cluster_point)
            self.forecast()

        # adding dates
        dates = test_data.index.copy()
        n_test_dates = dates.shape[0]

        slc = slice(-n_test_dates - 1, -1)

        hist_output_new = Series(self._hist_output[slc], index=dates).copy()
        self.hist_output = concat([self.hist_output, hist_output_new]).copy()

        rules_outputs_hist_new = DataFrame.from_records(self._rules_outputs_hist[slc], index=dates).copy()
        self.rules_outputs_hist = concat([self.rules_outputs_hist, rules_outputs_hist_new], axis='index').copy()

        slc = slice(-n_test_dates, None)
        membership_degrees_hist_new = DataFrame.from_records(self._membership_degrees_hist[slc], index=dates).copy()
        self.membership_degrees_hist = concat([self.membership_degrees_hist, membership_degrees_hist_new],
                                              axis='index').copy()

        clusters_parameters_hist_new = DataFrame.from_records(self._clusters_parameters_hist[slc], index=dates).copy()
        self.clusters_parameters_hist = concat([self.clusters_parameters_hist, clusters_parameters_hist_new],
                                               axis='index').copy()
