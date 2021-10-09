import logging
from scipy.stats import multivariate_normal


def calc_membership_degree(x, membership_function_type: str = 'gaussian', parameters=None):
    """
    The function calculates membership degree of input ``x`` to cluster, characterized by ``parameters``
    :param x: 1d array-like; input vector
    :param membership_function_type: str; the name of a membership function to use; currently only 'gaussian'
    is implemented
    :param parameters: dict; parameters of a cluster, e.g. its center, variance, etc
    :return: float; membership degree
    """

    logger = logging.getLogger('calc_membership_degree')

    if membership_function_type == 'gaussian':
        center = parameters['center']
        cov_matrix = parameters['covariance_matrix']

        result = multivariate_normal(mean=center, cov=cov_matrix).pdf(x)
    else:
        logger.warning('Wrong membership function type or not yet implemented')
        return

    return result
