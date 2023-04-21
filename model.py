import logging
from typing import Union

from pandas import Series, DataFrame, concat
from numpy import array, diag, concatenate, arange, array_str, inf
from scipy.optimize import least_squares, differential_evolution
from IPython.core.debugger import set_trace

from clusterization import cluster_data, calc_gaussian_membership_degrees
from clusterization.ets import update_antecedent_part as ets_update_antecedent_part
from clusterization.all_methods import CLUSTERING_METHODS
from local_models import calc_cond_var_fuzzy
from local_models.garch import PAST_H_TYPE_DEFAULT, PAST_H_TYPES
from auxiliary import unpack_1d_parameters, pack_1d_parameters

module_logger = logging.getLogger(__name__)

OPTIMIZATION_ALGORITHMS = ['ls', 'differential evolution']


class FuzzyVolatilityModel:
    def __init__(self,
                 train_data: Series = None,
                 clusterization_method: str = 'gaussian',
                 membership_function: str = 'gaussian',
                 clusterization_parameters: dict = None,
                 local_method: str = 'garch',
                 local_method_parameters: dict = None,
                 data_to_cluster: Union[str, Series, DataFrame] = 'train',
                 n_last_points_to_use_for_clustering: int = None,
                 cluster_sets_conjunction: Union['str', callable] = 'prod',
                 n_cluster_sets: int = None,
                 normalize: bool = False,
                 n_points_fitting: int = None,
                 first_h: Series = None,
                 optimization: str = 'ls',
                 optimization_parameters: dict = None,
                 clustered_space_dim: int = None,
                 past_h_type: str = PAST_H_TYPE_DEFAULT
                 ):
        self.logger = logging.getLogger(module_logger.name + '.' + type(self).__name__)
        self.logger.info(f'Creating an instance of {self.logger.name}')

        # train data
        if train_data is None:
            self.train_data = Series(dtype=float).copy()
        else:
            self.train_data = train_data.copy()

        # antecedent metaparameters
        self.clusterization_method = clusterization_method
        self.clusterization_parameters = clusterization_parameters
        self.membership_function = membership_function

        # consequent metaparameters
        self.local_method = local_method
        self.local_method_parameters = local_method_parameters

        self.consequent_parameters_ini = self.local_method_parameters['parameters_ini']
        self.n_points_fitting = n_points_fitting
        self._fitting_slice = slice(-self.n_points_fitting if self.n_points_fitting is not None else None, None)
        self.bounds = self.local_method_parameters['bounds']
        self.p = self.local_method_parameters['p']
        self.q = self.local_method_parameters['q']
        self.starting_index = max(self.p, self.q)
        self.past_h_type = past_h_type
        if first_h is None:
            if self.past_h_type == 'aggregated':
                self._first_h_1d_current = self.train_data[:self.starting_index] ** 2
                self.first_h_current = self._first_h_1d_current.copy()
            elif self.past_h_type == 'rule-wise':
                raise ValueError(f"""If `past_h_type == 'rule-wise'`, `first_h` should be provided`""")
            else:
                raise ValueError(f'`past_h_type` should be one of {PAST_H_TYPES}; got {self.past_h_type}')
        else:
            self._first_h_1d_current = None
            self.first_h_current = first_h.copy()

        if self.n_points_fitting is not None and self.n_points_fitting > len(self.train_data):
            raise ValueError('`n_points_fitting` should not be greater than '
                             '`len(train_data) - max(p, q)`; '
                             f'got {self.n_points_fitting} > {len(self.train_data) - self.starting_index}')

        # clusters parameters
        self._clusters_parameters_hist = []
        self.clusters_parameters_hist = DataFrame(dtype=float).copy()
        self.clusters_parameters_current = None

        self._n_clusters_hist = []
        self.n_clusters_hist = DataFrame(dtype=float).copy()
        self.n_clusters = None

        self.clustered_space_dim = clustered_space_dim

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

        # clustering algorithm
        if self.clusterization_method == 'gaussian' or self.clusterization_method == 'trapezoidal' \
                or type(self.clusterization_method) is list:
            self.cluster = self._cluster
        elif self.clusterization_method == 'eTS':
            self.cluster = self._cluster_ets

            if self.membership_function != 'gaussian':
                raise ValueError(f'Membership function of form {self.membership_function} is not supported for '
                                 f'clustering method {self.clusterization_method}')

            if self.data_to_cluster is not None:
                self.clustered_space_dim = self.data_to_cluster.shape[1]
            elif self.clustered_space_dim is not None:
                pass
            else:
                self.clustered_space_dim = self.p + self.q
                self.logger.warning(f'Both `data_to_cluster` and `clustered_space_dim` are None; '
                                    f'`clustered_space_dim` is set to be equal to `p + q`')

            # initializing `clusters_parameters_current`
            self.clusters_parameters_current = self.clusterization_parameters.copy()

            clusters_variance = self.clusterization_parameters['variance']
            self.cov_matrix = diag([clusters_variance] * self.clustered_space_dim, k=0)
        elif self.clusterization_method == 'eClustering':
            raise NotImplementedError('eClustering antecedent learning is not implemented')
        else:
            raise ValueError(f'Clustering method name {self.clusterization_method} '
                             f'is wrong or method is not implemented; '
                             f'should be one of {CLUSTERING_METHODS}')

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

        self._hist_output_fuzzy: list = []

        # consequent parameters
        self.alpha_0 = None
        self.alpha = None
        self.beta = None
        self._parameters_hist = []
        self._ls_results_hist = []

        # GARCH variables
        self.h = None
        self._h_hist = []

        self.h_fuzzy: array = None
        self._h_fuzzy_hist: list = []

        # optimization algorithm
        self.optimization = optimization

        if optimization == 'ls':
            self._fit = self._fit_ls
        elif optimization == 'differential evolution':
            self._fit = self._fit_de
            self.bounds = array(self.bounds).T  # scipy's `differential_evolution` takes bounds in different form
        else:
            raise ValueError(f'Optimization algorithms other than {OPTIMIZATION_ALGORITHMS} are not supported; '
                             f'got {optimization}')

        if optimization_parameters is None:
            self.optimization_parameters = {}
        else:
            self.optimization_parameters = optimization_parameters

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

    def _calc_residuals(self, _parameters):
        alpha_0, alpha, beta = unpack_1d_parameters(_parameters, p=self.p, q=self.q, n_clusters=self.n_clusters)

        h = calc_cond_var_fuzzy(alpha_0, alpha, beta, self.train_data[self._fitting_slice] ** 2, self.first_h_current,
                                weights=self.membership_degrees_current, past_h_type=self.past_h_type)

        residuals = self.train_data[self._fitting_slice][self.starting_index:] ** 2 - h[self.starting_index:-1]
        self.logger.debug(f'residuals = {residuals.tolist()}')
        self.logger.debug(f'RSS = {(residuals ** 2).sum()}')

        return residuals

    def _calc_rss(self, _parameters):
        return (self._calc_residuals(_parameters) ** 2).sum()

    def _cluster(self):
        self.logger.debug('Starting clusterization')

        clusterization_result = \
            cluster_data(self.data_to_cluster,
                         methods=self.clusterization_method,
                         parameters=self.clusterization_parameters,
                         n_last_points_to_use_for_clustering=self.n_last_points_to_use_for_clustering,
                         conjunction=self.cluster_sets_conjunction,
                         n_sets=self.n_cluster_sets,
                         normalize=self.normalize)

        self.clusters_parameters_current = clusterization_result['parameters'].copy()
        self.n_clusters = self.clusters_parameters_current['n_clusters']

        self.membership_degrees_current = clusterization_result['membership']

        self.logger.debug(f'Clusterization completed\n'
                          f'Estimated parameters: {self.clusters_parameters_current}\n'
                          f'Membership degrees:\n{self.membership_degrees_current}')

        self._clusters_parameters_hist.append(self.clusters_parameters_current)
        self._membership_degrees_hist.append(self.membership_degrees_current)
        self._n_clusters_hist.append(self.n_clusters)

    def _cluster_ets(self):
        if type(self.data_to_cluster) is not DataFrame:
            raise ValueError(f'`x` should be a pandas.DataFrame; got `type(x)` = {type(self.data_to_cluster)}')
        if self.n_last_points_to_use_for_clustering is not None:
            raise ValueError(f'For clustering method {self.clusterization_method} '
                             f'`n_last_points_to_use_for_clustering` should be None; '
                             f'got {self.n_last_points_to_use_for_clustering}')

        sigma_prev = self.clusters_parameters_current['sigma']
        beta_prev = self.clusters_parameters_current['beta']
        potentials_focal_prev = self.clusters_parameters_current['potentials_focal']
        delta_min = self.clusters_parameters_current['delta_min']
        focals_current = self.clusters_parameters_current['centers']

        t = self.data_to_cluster.shape[0]

        sigma_new, beta_new, focals_new, potentials_focal_new = \
            ets_update_antecedent_part(sigma_prev,
                                       beta_prev,
                                       self.cov_matrix[0, 0],
                                       focals_current,
                                       potentials_focal_prev,
                                       x_prev=self.data_to_cluster.iloc[-2].values,
                                       x_new=self.data_to_cluster.iloc[-1].values,
                                       t=t,
                                       delta_min=delta_min,
                                       )

        parameters_new = {
            'centers': focals_new,
            'sigma': sigma_new,
            'beta': beta_new,
            'focals': focals_new,
            'potentials_focal': potentials_focal_new,
            'n_clusters': len(focals_new)
        }

        self.clusters_parameters_current = self.clusters_parameters_current.copy()
        self.clusters_parameters_current.update(parameters_new)

        if self.n_clusters != len(focals_new):
            self.n_clusters = len(focals_new)

            _new_cluster_alpha_0_ini = self.consequent_parameters_ini['alpha_0'].mean()
            self.consequent_parameters_ini['alpha_0'] = \
                concatenate((self.consequent_parameters_ini['alpha_0'], [_new_cluster_alpha_0_ini])).copy()

            _new_cluster_alpha_ini = self.consequent_parameters_ini['alpha'].mean(axis=1)
            self.consequent_parameters_ini['alpha'] = \
                concatenate((self.consequent_parameters_ini['alpha'], array([_new_cluster_alpha_ini]).T), axis=1).copy()

            _new_cluster_beta_ini = self.consequent_parameters_ini['beta'].mean(axis=1)
            self.consequent_parameters_ini['beta'] = \
                concatenate((self.consequent_parameters_ini['beta'], array([_new_cluster_beta_ini]).T), axis=1).copy()

            if self.optimization == 'ls':
                self.bounds = self._add_bound(self.bounds, self.n_clusters)
            elif self.optimization == 'differential evolution':
                self.bounds = self._add_bound(self.bounds.T, self.n_clusters).T.copy()
            else:
                raise NotImplementedError(f'bounds recalculation for optimization of type {self.optimization} '
                                          f'is not implemented')

            bounds_str = array_str(self.bounds, max_line_width=inf).replace('\n', '')
            self.logger.debug(f"""new bounds = {bounds_str}""")

        cov_matrices = [self.cov_matrix for _ in range(self.n_clusters)]

        self.membership_degrees_current = calc_gaussian_membership_degrees(
            self.data_to_cluster.iloc[-1],
            focals_new,
            cov_matrices
        )

        self._clusters_parameters_hist.append(self.clusters_parameters_current)
        self._membership_degrees_hist.append(self.membership_degrees_current)
        self._n_clusters_hist.append(self.n_clusters)

    @staticmethod
    def _add_bound(bounds: Union[array, tuple],
                   n_clusters: int):
        """

        :param bounds: 2D array-like w/ 2 rows, first row representing lower bounds, and second row - upper bounds
        :param n_clusters: int, number of clusters
        :return:
        """

        bounds_new = array(
            [
                array(
                    [
                        list(_bounds_ul[i:i + n_clusters - 1]) + [_bounds_ul[i + n_clusters - 2]]
                        for i in arange(0, len(_bounds_ul), n_clusters - 1)
                    ]
                ).flatten()
                for _bounds_ul in bounds
            ]
        ).copy()

        return bounds_new

    def fit(self, train_data: Series = None, n_points: int = None):
        if train_data is not None:
            self.train_data = train_data.copy()

        _fitting_slice_ini = self._fitting_slice
        self._fitting_slice = slice(-n_points if n_points is not None else None, None)

        self._fit()

        # resetting fitting slice back to what it was at initialization
        self._fitting_slice = _fitting_slice_ini

    def _fit_ls(self):
        self.logger.debug('Starting fitting')

        parameters_0 = pack_1d_parameters(self.consequent_parameters_ini['alpha_0'],
                                          self.consequent_parameters_ini['alpha'],
                                          self.consequent_parameters_ini['beta'])

        self.logger.debug(f'Starting least squares estimation of parameters; `parameters_0`: {parameters_0}')
        ls_result = least_squares(self._calc_residuals, parameters_0, bounds=self.bounds,
                                  **self.optimization_parameters)
        self._ls_results_hist.append(ls_result)

        parameters = ls_result.x
        self.logger.debug(f'Least squares estimation finished; estimated parameters = {parameters}, '
                          f'LS results: {ls_result}')

        self.alpha_0, self.alpha, self.beta = \
            unpack_1d_parameters(parameters, p=self.p, q=self.q, n_clusters=self.n_clusters)
        self._parameters_hist.append({'alpha_0': self.alpha_0, 'alpha': self.alpha, 'beta': self.beta})
        self.consequent_parameters_ini = self._parameters_hist[-1].copy()

        self.logger.debug('Fitting is completed')

    def _fit_de(self):
        self.logger.debug('Starting fitting')

        parameters_0 = pack_1d_parameters(self.consequent_parameters_ini['alpha_0'],
                                          self.consequent_parameters_ini['alpha'],
                                          self.consequent_parameters_ini['beta'])

        ls_result = differential_evolution(self._calc_rss, bounds=self.bounds, x0=parameters_0,
                                           **self.optimization_parameters)
        self._ls_results_hist.append(ls_result)

        parameters = ls_result.x
        self.alpha_0, self.alpha, self.beta = \
            unpack_1d_parameters(parameters, p=self.p, q=self.q, n_clusters=self.n_clusters)

        self._parameters_hist.append({'alpha_0': self.alpha_0, 'alpha': self.alpha, 'beta': self.beta})
        self.consequent_parameters_ini = self._parameters_hist[-1].copy()

        self.logger.debug('Fitting is completed')

    def forecast(self, n_points: int = None):
        _fitting_slice_ini = self._fitting_slice
        self._fitting_slice = slice(-n_points if n_points is not None else None, None)

        self._forecast()

        # resetting fitting slice back to what it was at initialization
        self._fitting_slice = _fitting_slice_ini

    def _forecast(self):
        self.h_fuzzy, self.h = calc_cond_var_fuzzy(self.alpha_0, self.alpha, self.beta,
                                                   self.train_data[self._fitting_slice] ** 2, self.first_h_current,
                                                   return_fuzzy=True,
                                                   weights=self.membership_degrees_current,
                                                   past_h_type=self.past_h_type)

        self._h_hist.append(self.h)
        self.current_output = self.h[-1]
        self._hist_output.append(self.current_output)

        if self.past_h_type == 'rule-wise':
            self._h_fuzzy_hist.append(self.h_fuzzy)
            self._hist_output_fuzzy.append(self.h_fuzzy[-1, :])

    def _push(self, observation: float, observation_date, data_to_cluster_point):
        self.train_data.loc[observation_date] = observation
        if type(self.data_to_cluster) is not str:
            self.data_to_cluster.loc[observation_date] = data_to_cluster_point
        elif self.data_to_cluster != 'train':
            raise ValueError("""`data_to_cluster` should be either a string 'train' or not a string""")

        self._first_h_1d_current = self.train_data[self._fitting_slice][:self.starting_index] ** 2

        self.cluster()
        self._set_first_h()
        self._fit()

    def _set_first_h(self):
        if self.past_h_type == 'aggregated':
            self.first_h_current = self._first_h_1d_current.copy()
        elif self.past_h_type == 'rule-wise':
            self.first_h_current = array([self._first_h_1d_current.copy() for _ in range(self.n_clusters)]).T.copy()
        else:
            raise ValueError(f'`past_h_type` should be one of {PAST_H_TYPES}; got {self.past_h_type}')

    def feed_daily_data(self, test_data: Series, data_to_cluster=None, n_points: int = None):
        if self.current_output is None:
            # if there is no current forecast (i.e., only initial fitting was performed),
            # then do forecast before running the main algorithm

            _fitting_slice_ini = self._fitting_slice
            self._fitting_slice = slice(-n_points if n_points is not None else None, None)

            self.forecast(n_points=n_points)

            # resetting fitting slice back to what it was at initialization
            self._fitting_slice = _fitting_slice_ini

        data_to_cluster = self._convert_data_to_cluster(data_to_cluster,
                                                        test_data,
                                                        variable_name='self.data_to_cluster')
        if (len(data_to_cluster.shape) != len(self.data_to_cluster.shape)) or \
                (len(data_to_cluster.shape) == len(self.data_to_cluster.shape) == 2 and
                 data_to_cluster.shape[1] != self.data_to_cluster.shape[1]):
            raise ValueError('# of columns is different in given `data_to_cluster` and `self.data_to_cluster`')

        # imitating live daily algorithm work
        for date in test_data.index:
            self.logger.debug(f'feed_daily_data: new iteration @ {date}')

            observation = test_data.loc[date]
            data_to_cluster_point = data_to_cluster.loc[date]
            self._push(observation, date, data_to_cluster_point)
            self._forecast()

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

    def show_ls_results(self):
        _attrs = ['cost', 'optimality', 'nfev', 'njev', 'status', 'message', 'success']
        ls_res = DataFrame.from_records([{_attr: _ls_res[_attr] for _attr in _attrs}
                                         for _ls_res in self._ls_results_hist]).copy()

        return ls_res

    def show_antecedent_hist(self):
        return DataFrame.from_records(self._clusters_parameters_hist).copy()
