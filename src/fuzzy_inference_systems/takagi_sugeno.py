import logging

from numpy import array, concatenate, nan, inf, array_str, eye
from scipy.optimize import least_squares

from src.clusterization import calc_gaussian_membership_degrees
from src.consequent_estimators import re_estimate_params_plain_rls, ets_new_cluster_re_estimate_parameters
from src.rules_related import combine_rules_outputs
from src.clusterization.ets import update_antecedent_part as ets_update_antecedent_part

OPTIMIZATION_ALGORITHMS = ('ls',)
CLUSTERING_METHODS = ('eTS', 'eClustering', 'FCM')


class TSFuzzyInferenceSystemBase:
    def __init__(self,
                 membership_function: str = 'gaussian',
                 clusters_params: dict = None,
                 data_to_cluster: array = None,
                 normalize: bool = False,
                 ):
        """

        :param membership_function:
        :param clusters_params:
        :param data_to_cluster:
        :param normalize:
        """
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.info(f'Creating an instance of {self.logger.name}')

        self.input: array = None

        # antecedent
        self.n_clusters: int = nan
        self._n_clusters_hist: list = []

        self.normalize: bool = normalize

        self.data_to_cluster: array = data_to_cluster
        self._data_to_cluster_hist: list = [data_to_cluster]

        self.clusters_params: dict = clusters_params
        self._clusters_params_hist: list = [self.clusters_params]

        self.membership_degrees: array = None
        self._membership_degrees_hist: list = [self.membership_degrees]

        if membership_function == 'gaussian':
            self.calc_membership_degrees = self._calc_gaussian_membership_degrees

        # consequent
        self.consequent_params: array = None
        self._consequent_params_hist: list = []

        # outputs
        self.fuzzy_output: array = None
        self.fuzzy_output_hist: list = []
        self.output: float = nan
        self.output_hist: list = []

    def _calc_gaussian_membership_degrees(self):
        self.membership_degrees = calc_gaussian_membership_degrees(self.data_to_cluster,
                                                                   centers=self.clusters_params['centers'],
                                                                   cov_matrices=self.clusters_params['cov_matrices'],
                                                                   normalize=self.normalize)

    @staticmethod
    def _calc_consequent(params: array,
                         input_variables: array
                         ) -> float:
        consequent = params[0] + (params[1:] * input_variables).sum()
        return consequent

    @staticmethod
    def _calc_consequents(params: array,
                          regressors: array
                          ) -> array:
        input_ext = concatenate([[1], regressors]).copy()
        fuzzy_output = params @ input_ext

        return fuzzy_output

    def calc_consequents(self, add_to_hist: bool = True):
        """
        Calculate fuzzy outputs (output of each fuzzy rule)
        :param add_to_hist: bool. Whether to add an output to historical array
        """
        self.fuzzy_output = self._calc_consequents(self.consequent_params, self.input).copy()

        if add_to_hist:
            self.fuzzy_output_hist.append(self.fuzzy_output)

    @staticmethod
    def _calc_consequents_hist(params: array,
                               regressors_hist: array
                               ) -> array:
        """

        :param params: 2D numpy.array. Each row is a single cluster's parameters.
        :param regressors_hist: 2D numpy.array. Each row is a certain time point
        :return: 2D numpy.array. Each row is a fuzzy output for a certain time point
        """
        hist_depth = regressors_hist.shape[0]
        input_ext = concatenate(([[1] for _ in range(hist_depth)], regressors_hist), axis=1)
        return input_ext @ params.T

    def _defuzzify(self, fuzzy_output: array) -> float:
        """

        :param fuzzy_output:
        :return:
        """
        return combine_rules_outputs(fuzzy_output, self.membership_degrees)

    def _defuzzify_hist(self, fuzzy_output_hist: array) -> array:
        """
        Aggregate historical fuzzy output into historical crisp output GIVEN the fixed membership degrees
        :param fuzzy_output_hist: 2D numpy.array. Historical fuzzy output. Each row is a certain time point
        :return: 1D numpy.array. Aggregated (defuzzified crisp) output for each time point
        """
        return fuzzy_output_hist @ self.membership_degrees / self.membership_degrees.sum()

    def defuzzify(self, add_to_hist: bool = True):
        """
        A defuzzification interface. Aggregate fuzzy outputs into a single crisp output.
        :param add_to_hist: bool. Whether to add an output to historical array
        """
        self.output = combine_rules_outputs(self.fuzzy_output, self.membership_degrees)

        if add_to_hist:
            self.output_hist.append(self.output)

    def defuzzify_hist(self):
        self.output_hist += self._defuzzify_hist(self.fuzzy_output_hist)


class TSFuzzyInferenceSystem(TSFuzzyInferenceSystemBase):
    def __init__(self,
                 optimization: str = 'ls',
                 optimization_params: dict = None,
                 consequent_metaparams: dict = None,
                 clusterization_method: str = None,
                 clusterization_metaparams: dict = None,
                 n_last_pts_clustering: int = None,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

        # clustering
        self.clusterization_method: str = clusterization_method
        self.n_last_pts_clustering: int = n_last_pts_clustering

        self.clusterization_metaparams: dict = clusterization_metaparams
        if self.clusterization_metaparams is None:
            self.clusterization_metaparams = {}

        if self.clusterization_method == 'eTS':
            self.cluster: callable = self._cluster_ets

            if self.n_last_pts_clustering is not None:
                raise ValueError(f'For clustering method eTS'
                                 f'`n_last_pts_clustering` should be None; '
                                 f'got {self.n_last_pts_clustering}')
        elif self.clusterization_method == 'FCM':
            raise NotImplementedError('FCM algorithm is not implemented')
        elif self.clusterization_method == 'eClustering':
            raise NotImplementedError('eClustering antecedent learning is not implemented')
        else:
            raise ValueError(f'Clustering method name {self.clusterization_method} '
                             f'is wrong or method is not implemented; '
                             f'should be one of {CLUSTERING_METHODS}')

        self.rls_cov: array = None
        self._rls_cov_hist: list = []

        # consequent
        self.consequent_metaparams: dict = consequent_metaparams
        self.consequent_params_ini: array = self.consequent_metaparams['params_ini']

        if optimization == 'ls' or optimization == 'differential evolution':
            self.consequent_params_ini = self.consequent_metaparams['parameters_ini']
            self.bounds = self.consequent_metaparams['bounds']

        # optimization algorithm
        self.optimization = optimization

        if optimization == 'ls':
            self.fit = self._fit_ls
        # elif optimization == 'differential evolution':
        #     self.fit = self._fit_de
        #     self.bounds = array(self.bounds).T  # scipy's `differential_evolution` takes bounds in different form
        # elif optimization.lower() == 'rls':
        #     self.fit = self._fit_rls
        #
        #     self.rls_cov: array = None
        #     self._rls_cov_hist: list = []
        #
        #     self.rls_omega = rls_omega
        else:
            raise ValueError(f'Optimization algorithms other than {OPTIMIZATION_ALGORITHMS} are not supported; '
                             f'got {optimization}')

        if optimization_params is None:
            self.optimization_params = {}
        else:
            self.optimization_params = optimization_params

        # fit data
        self.actual_output_hist: array = None
        self.regressors_hist: array = None

    def _fit_ls(self):
        self.logger.debug('Starting least squares estimation of parameters; `parameters_0`: '
                          f'{self.consequent_params_ini}')

        bounds = self.bounds.flatten().copy()
        ls_result = least_squares(self._calc_residuals,
                                  self.consequent_params_ini,
                                  bounds=bounds,
                                  **self.optimization_params)
        # self._ls_results_hist.append(ls_result)

        self.consequent_params = ls_result.x
        self._consequent_params_hist.append(self.consequent_params)
        self.logger.debug(f'Least squares estimation finished; estimated parameters = {self.consequent_params}, '
                          f'LS results: {ls_result}')

        # self.consequent_params_ini = self._parameters_hist[-1].copy()

        self.logger.debug('Fitting is completed')

    def _calc_residuals(self, _params: array) -> array:
        _params = _params.reshape(self.n_clusters, -1).copy()

        forecast_fuzzy_hist = self._calc_consequents_hist(_params,
                                                          self.regressors_hist).copy()
        forecast_hist = self._defuzzify_hist(forecast_fuzzy_hist).copy()
        residuals = self.actual_output_hist - forecast_hist

        return residuals

    def _fit_rls(self):
        logger = logging.getLogger(self.logger.name + '._fit_rls')

        if self._n_clusters_hist[-2] == self._n_clusters_hist[-1]:
            # # of clusters didn't change
            logger.debug('# of clusters unchanged, using plain RLS for estimation')

            self.rls_cov, self.consequent_params = re_estimate_params_plain_rls(
                params_prev=self.consequent_params,
                cov_prev=self.rls_cov,
                y_new=self.input.iloc[-1],
                coeffs_prev=self.input
            )
        elif self._n_clusters_hist[-2] + 1 == self._n_clusters_hist[-1]:
            # # of clusters increased by 1
            logger.debug('# of clusters increased by 1, using eTS-based algorithm for estimation')

            rls_omega = self.consequent_metaparams['omega']

            self.rls_cov, self.consequent_params = ets_new_cluster_re_estimate_parameters(
                params_prev=self.consequent_params,
                cov_prev=self.rls_cov,
                weights=self.membership_degrees[:-1],
                n_params_in_a_rule=self.consequent_params.shape[0],
                omega=rls_omega
            )
        else:
            raise ValueError('RLS can only handle when # clusters increases by not more than 1; '
                             f'got prev # clusters = {self._n_clusters_hist[-2]}, '
                             f'new # clusters = {self._n_clusters_hist[-1]}')

        self._consequent_params_hist.append(self.consequent_params)
        self._rls_cov_hist.append(self.rls_cov)

    def _cluster_ets(self):
        sigma_prev = self.clusters_params['sigma']
        beta_prev = self.clusters_params['beta']
        spread = self.clusterization_metaparams['spread']
        potentials_focal_prev = self.clusters_params['potentials_focal']
        delta_min = self.clusters_params['delta_min']
        focals_current = self.clusters_params['centers']

        t = len(self._data_to_cluster_hist)

        sigma_new, beta_new, focals_new, potentials_focal_new = \
            ets_update_antecedent_part(sigma_prev,
                                       beta_prev,
                                       spread,
                                       focals_current,
                                       potentials_focal_prev,
                                       x_prev=self._data_to_cluster_hist[-2],
                                       x_new=self._data_to_cluster_hist[-1],
                                       t=t,
                                       delta_min=delta_min,
                                       )

        params_new = {
            'centers': focals_new,
            'sigma': sigma_new,
            'beta': beta_new,
            'focals': focals_new,
            'potentials_focal': potentials_focal_new,
            'n_clusters': len(focals_new)
        }

        self.clusters_params.update(params_new)

        if self.n_clusters != len(focals_new):
            self.n_clusters = len(focals_new)
            self.logger.info(f'new # of clusters: {self.n_clusters}')

            _new_params = self.consequent_params_ini.mean(axis=1)
            self.consequent_params_ini = \
                concatenate((self.consequent_params_ini, _new_params.reshape(-1, 1)), axis=1).copy()

            if self.optimization == 'ls' or self.optimization == 'differential evolution':
                self.bounds = self._add_bound(self.bounds)
            elif self.optimization.lower() == 'rls':
                pass
            else:
                raise NotImplementedError(f'bounds recalculation for optimization of type {self.optimization} '
                                          f'is not implemented')

            bounds_str = array_str(array(self.bounds), max_line_width=inf).replace('\n', '')
            self.logger.debug(f"""new bounds = {bounds_str}""")

            self.clusters_params['cov_matrices'] = [spread * eye(self.consequent_params.shape[1])
                                                    for _ in range(self.n_clusters)]

        self._calc_gaussian_membership_degrees()

        self._clusters_params_hist.append(self.clusters_params)
        self._membership_degrees_hist.append(self.membership_degrees)
        self._n_clusters_hist.append(self.n_clusters)

    @staticmethod
    def _add_bound(bounds: tuple) -> tuple:
        """

        :param bounds: tuple of length 2. Each element is a 2D numpy.array representing a lower or upper bounds
        for consequent parameters. Each numpy.array is of shape (n_clusters, p), where p is # of consequent
        params in a single cluster
        :return: same as `bounds` but each matrix is augmented with a row in the end for a new cluster
        """

        bounds_new = [
            concatenate((_bounds_ul, _bounds_ul[-1, :].reshape(1, -1)), axis=0)
            for _bounds_ul in bounds
        ]
        bounds_new = tuple(bounds_new)

        return bounds_new
