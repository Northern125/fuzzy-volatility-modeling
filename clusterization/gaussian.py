import logging
from scipy.stats import multivariate_normal
from numpy import array


def calc_gaussian_membership_degrees(input_data, centers, cov_matrices):
    logger = logging.getLogger('calc_gaussian_membership_degrees')

    logger.info('Starting')

    result = [multivariate_normal(mean=center, cov=cov_matrix).pdf(input_data)
              for center, cov_matrix in zip(centers, cov_matrices)]
    result = array(result)

    return result
