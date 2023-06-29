import logging

from numpy import array, concatenate, nan
from scipy.optimize import least_squares

from src.clusterization import calc_gaussian_membership_degrees
from src.rules_related import combine_rules_outputs

OPTIMIZATION_ALGORITHMS = ['ls']


class TSFuzzyInferenceSystem:
    def __init__(self,
                 membership_function: str = 'gaussian',
                 clusters_params: dict = None,
                 data_to_cluster: array = 'train',
                 normalize: bool = False,
                 optimization: str = 'ls',
                 optimization_params: dict = None,
                 consequent_metaparams: dict = None,
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
        self.normalize: bool = normalize
        self.data_to_cluster: array = data_to_cluster
        self.clusters_params: dict = clusters_params

        if membership_function == 'gaussian':
            self.calc_membership_degrees = self._calc_gaussian_membership_degrees

        # consequent
        self.consequent_params: array = None
        self.consequent_metaparams = consequent_metaparams

        if optimization == 'ls' or optimization == 'differential evolution':
            self.consequent_params_ini = self.consequent_metaparams['parameters_ini']
            self.bounds = self.consequent_metaparams['bounds']

        # outputs
        self.fuzzy_output: array = None
        self.fuzzy_output_hist: list = []
        self.output: float = nan
        self.output_hist: list = []

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
        fuzzy_output = (params @ input_ext).copy()

        return fuzzy_output

    def calc_consequents(self, add_to_hist: bool = True):
        """
        Calculate fuzzy outputs (output of each fuzzy rule)
        :param add_to_hist: bool. Whether or not to add an output to historical array
        """
        input_ext = concatenate([[1], self.input]).copy()
        self.fuzzy_output = (self.consequent_params @ input_ext).copy()

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
        :param add_to_hist: bool. Whether or not to add an output to historical array
        """
        self.output = combine_rules_outputs(self.fuzzy_output, self.membership_degrees)

        if add_to_hist:
            self.output_hist.append(self.output)

    def _fit_ls(self):
        self.logger.debug('Starting least squares estimation of parameters; `parameters_0`: '
                          f'{self.consequent_params_ini}')
        ls_result = least_squares(self._calc_residuals,
                                  self.consequent_params_ini,
                                  bounds=self.bounds,
                                  **self.optimization_params)
        # self._ls_results_hist.append(ls_result)

        self.consequent_params = ls_result.x
        # self._parameters_hist.append({'alpha_0': self.alpha_0, 'alpha': self.alpha, 'beta': self.beta})
        self.logger.debug(f'Least squares estimation finished; estimated parameters = {self.consequent_params}, '
                          f'LS results: {ls_result}')

        # self.consequent_parameters_ini = self._parameters_hist[-1].copy()

        self.logger.debug('Fitting is completed')

    def _calc_residuals(self, _params: array) -> array:
        forecast_fuzzy_hist = self._calc_consequents_hist(_params,
                                                          self.regressors_hist).copy()
        forecast_hist = self._defuzzify_hist(forecast_fuzzy_hist).copy()
        residuals = self.actual_output_hist - forecast_hist

        return residuals
