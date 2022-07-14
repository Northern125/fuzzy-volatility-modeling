import logging
from typing import Union
from numpy import diag, array
from pandas import Series
from itertools import product

from .gaussian import calc_gaussian_membership_degrees
from membership_functions import calc_trapezoidal_membership_degrees


def cluster_data(x: list,
                 methods: list,
                 parameters: list = None,
                 n_last_points_to_use_for_clustering: list = None,
                 conjunction: Union[str, callable] = 'prod') -> dict:
    """
    Cluster data given different cluster sets, combine them via Cartesian product and then perform conjunction
    (conjunction method is passed to `conjunction`). For example, if 2 cluster sets are given, first of size 2
    and second of size 3, the resulting # of clusters will be 6.
    :param x: 2D array-like, input data to cluster for each cluster set
    :param methods: 1D list of `str`, method for each cluster set
    :param parameters: 1D list of `dict`, parameters of the clusterization algorithm (e.g., number of clusters, etc)
    for each cluster set
    :param n_last_points_to_use_for_clustering: 1D list of `int`, number of last points to use in `x[i]` for clustering
    for each i-th cluster set
    :param conjunction: `str` in ('prod') or `callable`, conjunction method for combining membership degrees from
    different sets. Default 'prod' -- product
    :return: `dict` {'parameters': estimated parameters of clusters, 'membership': 1D array of membership degrees
    of all `x`'s to resulting clusters}
    """
    logger = logging.getLogger('cluster_data')

    # workaround for backward compatibility (if given variables are 1D)
    if ((type(x) is array or type(x) is Series) and len(x.shape) == 1) or \
            (type(x) is list and len(array(x).shape) == 1):
        logger.debug('`x` is a 1D array, putting it and other variables into lists')
        x = [x]
        methods = [methods]
        if parameters is not None:
            parameters = [parameters]
        if n_last_points_to_use_for_clustering is not None:
            n_last_points_to_use_for_clustering = [n_last_points_to_use_for_clustering]

    # checking lengths
    if len(x) != len(methods):
        raise ValueError(f'Lengths of `x` and `methods` should coincide. Got `len(x) = {len(x)}`, '
                         f'`len(methods) = {len(methods)}`')
    if parameters is not None and len(x) != len(parameters):
        raise ValueError(f'Lengths of `x` and `parameters` should coincide. Got `len(x) = {len(x)}`, '
                         f'`len(parameters) = {len(parameters)}`')
    if n_last_points_to_use_for_clustering is not None and len(x) != len(n_last_points_to_use_for_clustering):
        raise ValueError(f'Lengths of `x` and `parameters` should coincide. Got `len(x) = {len(x)}`, '
                         f'`len(n_last_points_to_use_for_clustering) = {len(n_last_points_to_use_for_clustering)}`')

    logger.info(f'# of sets of clusters is {len(x)}')

    # performing clustering separately
    clustering_results = []
    for i, (_x, _method) in enumerate(zip(x, methods)):
        _clustering_result = cluster_data_1d(_x,
                                             method=_method,
                                             parameters=parameters[i] if parameters is not None else None,
                                             n_last_points_to_use_for_clustering=n_last_points_to_use_for_clustering[i]
                                             if n_last_points_to_use_for_clustering is not None else None)
        clustering_results.append(_clustering_result)

    logger.debug(f'Sets-wise clustering completed; `clustering_results`: {clustering_results}')

    # performing conjunction
    _membership_degrees = [_res['membership'] for _res in clustering_results]
    membership_degrees = list(product(*_membership_degrees, repeat=1))
    membership_degrees = [array(_tuple) for _tuple in membership_degrees]
    if conjunction == 'prod':
        membership_degrees = [_arr.prod() for _arr in membership_degrees]
    elif callable(conjunction):
        membership_degrees = [conjunction(_arr) for _arr in membership_degrees]
    else:
        raise NotImplementedError(f'Conjunction type {conjunction} is not implemented or its name is wrong')
    membership_degrees = array(membership_degrees)

    parameters_combined = [_res['parameters'] for _res in clustering_results]
    parameters_final = {'n_clusters': membership_degrees.shape[0], 'params by sets': parameters_combined}

    return {'parameters': parameters_final, 'membership': membership_degrees}


def cluster_data_1d(x: Union[list, array, Series],
                    method: str = 'gaussian',
                    parameters: dict = None,
                    n_last_points_to_use_for_clustering: int = None) -> dict:
    """
    Cluster data `x` and calculate membership degrees of `x` to different clusters
    :param x: 1D array-like, input data to cluster
    :param method: str, clustering method
    :param parameters: dict, parameters of the clusterization algorithm (e.g., number of clusters, etc)
    :param n_last_points_to_use_for_clustering: int, number of last points to use in `x` for clustering
    :return: `dict` {'parameters': estimated parameters of clusters, 'membership': 1D array of membership degrees
    of `x` to clusters}
    """

    logger = logging.getLogger('cluster_data_1d')

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
