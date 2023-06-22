import logging
from src.multivariate_normal_distribution import LongMultivariateNormal
from numpy import array


def calc_gaussian_membership_degrees(input_data, centers, cov_matrices, normalize: bool = False):
    logger = logging.getLogger('calc_gaussian_membership_degrees')

    logger.debug('Starting')

    result = [LongMultivariateNormal(mean=center, cov=cov_matrix).pdf(input_data)
              for center, cov_matrix in zip(centers, cov_matrices)]
    result = array(result).copy()

    if normalize:
        result /= result.sum()

    return result
