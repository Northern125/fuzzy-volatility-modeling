import logging
from numpy import diag

from .gaussian import calc_gaussian_membership_degrees


def cluster_data(x, method: str = 'gaussian', parameters: dict = None):
    """

    :param x: input
    :param method: clustering method
    :param parameters: parameters of the clusterization algorithm
    :return:
    """

    logger = logging.getLogger('cluster_input')

    n = x.shape[0]

    if method == 'gaussian':
        logger.debug('clustering method is gaussian')

        if parameters is not None:
            logger.debug('parameters is not None')

            centers = parameters['centers']
            variances = parameters['variances']

            centers = [[center] * n for center in centers]
            cov_matrices = [diag([variance] * n, k=0) for variance in variances]

            membership_degrees = calc_gaussian_membership_degrees(x, centers, cov_matrices)

            result = {'parameters': parameters, 'membership': membership_degrees}
        else:
            logger.debug('parameters is None')
            logger.warning('Algorithm for automatic Gaussian clusterization is not yet implemented. Exiting')
            return
    else:
        logger.warning('Clustering method name is wrong or method not yet implemented')
        return

    return result
