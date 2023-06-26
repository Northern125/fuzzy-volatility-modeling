import logging

from numpy import array, concatenate, matmul, nan

from src.clusterization import calc_gaussian_membership_degrees
from src.rules_related import combine_rules_outputs


class TSFuzzyInferenceSystem:
    def __init__(self,
                 membership_function: str = 'gaussian',
                 clusters_params: dict = None,
                 data_to_cluster: array = 'train',
                 normalize: bool = False,
                 ):
        """

        :param membership_function:
        :param clusters_params:
        :param data_to_cluster:
        :param normalize:
        """
        self.input: array = None

        # antecedent
        self.normalize: bool = normalize
        self.data_to_cluster: array = data_to_cluster
        self.clusters_params: dict = clusters_params

        if membership_function == 'gaussian':
            self.calc_membership_degrees = self._calc_gaussian_membership_degrees

        # consequent
        self.consequent_params: array = None

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

    def calc_consequents(self, add_to_hist: bool = True):
        """
        Calculate fuzzy outputs (output of each fuzzy rule)
        :param add_to_hist: bool. Whether or not to add an output to historical array
        """
        input_ext = concatenate([[1], self.input]).copy()
        self.fuzzy_output = matmul(self.consequent_params, input_ext).copy()

        if add_to_hist:
            self.fuzzy_output_hist.append(self.fuzzy_output)

    def defuzzify(self, add_to_hist: bool = True):
        """
        A defuzzification interface. Aggregate fuzzy outputs into a single crisp output.
        :param add_to_hist: bool. Whether or not to add an output to historical array
        """
        self.output = combine_rules_outputs(self.fuzzy_output, self.membership_degrees)

        if add_to_hist:
            self.output_hist.append(self.output)
