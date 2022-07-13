import logging
from typing import Union
from numpy import diag, array
from pandas import Series

from .gaussian import calc_gaussian_membership_degrees
from membership_functions import calc_trapezoidal_membership_degrees


def cluster_data(x: Union[list, array, Series], method: str = 'gaussian',
                 parameters: dict = None,
                 n_last_points_to_use_for_clustering: int = None) -> dict:
    """
    Cluster data `x` and calculate membership degrees of `x` to different clusters
    :param x: 1D array-like, input data to cluster
    :param method: str, clustering method
    :param parameters: dict, parameters of the clusterization algorithm (e.g., number of clusters, etc)
    :param n_last_points_to_use_for_clustering: int, number of last points to use in `x` for clustering
    :return: dict {'parameters': estimated parameters of clusters, 'membership': 1D array of membership degrees
    of `x` to clusters}
    """

    logger = logging.getLogger('cluster_data')

    n = x.shape[0]

    slc = slice(-n_last_points_to_use_for_clustering if n_last_points_to_use_for_clustering is not None else None, None)

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
            raise NotImplementedError('Algorithm for automatic Gaussian clusterization is not yet implemented '
                                      '(raised because `parameters` is None)')
    elif method == 'trapezoidal':
        logger.debug('clustering method is trapezoidal')

        if parameters is not None:
            a = parameters['a']
            b = parameters['b']
            c = parameters['c']
            d = parameters['d']

            membership_degrees = calc_trapezoidal_membership_degrees(x[slc], a, b, c, d)
            result = {'parameters': parameters, 'membership': membership_degrees}
        else:
            raise NotImplementedError('Algorithm for automatic trapezoidal clusterization is not implemented '
                                      '(raised because `parameters` is None)')
    else:
        raise Exception('Clustering method name is wrong or method is not implemented')

    return result
