import logging
from typing import Union
from numpy import diag, array
from pandas import Series, DataFrame
from itertools import product

from .ets import update_antecedent_part
from .gaussian import calc_gaussian_membership_degrees
from membership_functions import calc_trapezoidal_membership_degrees

CLUSTERING_METHODS = ('gaussian', 'trapezoidal', 'eTS', 'eClustering')


def cluster_data(x: Union[list, DataFrame],
                 methods: list,
                 membership_functions: list = None,
                 parameters: list = None,
                 n_last_points_to_use_for_clustering: list = None,
                 conjunction: Union[str, callable] = 'prod',
                 n_sets: int = None,
                 normalize: bool = False,
                 clusterization_parameters: list = None,
                 ) -> dict:
    """
    Cluster data given different cluster sets, combine them via Cartesian product and then perform conjunction
    (conjunction method is passed to `conjunction`). For example, if 2 cluster sets are given, first of size 2
    and second of size 3, the resulting # of clusters will be 6.

    :param x: 2D array-like, input data to cluster for each cluster set
    :param methods: 1D list of `str`, method for each cluster set
    :param membership_functions: 1D list of str, name of the membership function (e.g., 'gaussian') for each cluster set
    :param parameters: 1D list of `dict`, parameters of clusters (e.g., number of clusters, centers, variances, etc)
    for each cluster set
    :param n_last_points_to_use_for_clustering: 1D list of `int`, number of last points to use in `x[i]` for clustering
    for each i-th cluster set
    :param conjunction: `str` in ('prod') or `callable`, conjunction method for combining membership degrees from
    different sets. Default 'prod' -- product
    :param n_sets: int, optional, the number of cluster sets. If None, then inferred from `x` shape
    :param normalize: bool, whether to normalize the membership degrees in each cluster set so that their sum equals 1.
    I.e., whether to divide each membership degree by the sum of membership degrees INSIDE OF EACH CLUSTER SET. Setting
    this to `True` will actually yield final membership degrees (conjunct), which sum is also equal to 1
    :param clusterization_parameters: 1D list of dict, parameters of the clustering algorithm for each cluster set

    :return: `dict` {'parameters': estimated parameters of clusters, 'membership': 1D array of membership degrees
    of all `x`'s to resulting clusters}
    """
    logger = logging.getLogger('cluster_data')

    if membership_functions is None:
        membership_functions = ['gaussian', 'trapezoidal']

    x = x.copy()

    # workaround for backward compatibility (if given variables are 1D)
    if ((type(x) is array or type(x) is Series) and len(x.shape) == 1) or \
            (type(x) is list and array(x).dtype != object and len(array(x).shape) == 1) or \
            n_sets == 1:
        logger.debug('`x` is a 1D array, putting it and other variables into lists')
        x = [x].copy()
        methods = [methods]
        if parameters is not None:
            parameters = [parameters]
        if n_last_points_to_use_for_clustering is not None:
            n_last_points_to_use_for_clustering = [n_last_points_to_use_for_clustering]
        if membership_functions is not None:
            membership_functions = [membership_functions]
        if clusterization_parameters is not None:
            clusterization_parameters = [clusterization_parameters]

    # inferring `n_sets` if not given and checking length of `x` otherwise
    _n_sets_inferred = len(x) if type(x) is not DataFrame else x.shape[1]
    if n_sets is None:
        n_sets = _n_sets_inferred
        logger.info(f'Inferred # of sets of clusters is {n_sets}')
    elif _n_sets_inferred != n_sets:
        raise ValueError(f'Inferred from `x` # of sets not equal to `n_sets`. '
                         f'Got inferred # of sets = {_n_sets_inferred}, `n_sets = {n_sets}`')

    # checking lengths
    if n_sets != len(methods):
        raise ValueError(f'Lengths of `x` and `methods` should coincide. Got `len(x) = {len(x)}`, '
                         f'`len(methods) = {len(methods)}`')
    if parameters is not None and n_sets != len(parameters):
        raise ValueError(f'Lengths of `x` and `parameters` should coincide. Got `len(x) = {len(x)}`, '
                         f'`len(parameters) = {len(parameters)}`')
    if n_last_points_to_use_for_clustering is not None and n_sets != len(n_last_points_to_use_for_clustering):
        raise ValueError(f'Lengths of `x` and `parameters` should coincide. Got `len(x) = {len(x)}`, '
                         f'`len(n_last_points_to_use_for_clustering) = {len(n_last_points_to_use_for_clustering)}`')

    # replacing `parameters` & `n_last_points_to_use_for_clustering` if they are None
    if parameters is None:
        parameters = [None for _ in range(n_sets)]
    if n_last_points_to_use_for_clustering is None:
        n_last_points_to_use_for_clustering = [None for _ in range(n_sets)]
    if membership_functions is None:
        membership_functions = [None for _ in range(n_sets)]
    if clusterization_parameters is None:
        clusterization_parameters = [None for _ in range(n_sets)]

    # performing clustering separately
    if type(x) is DataFrame:
        x = [x.iloc[:, _] for _ in range(n_sets)]

    clustering_results = \
        [
            cluster_data_1d(_x,
                            method=_method,
                            membership_function=_membership_function,
                            parameters=_parameters,
                            n_last_points_to_use_for_clustering=_n_last_points_to_use_for_clustering,
                            normalize=normalize,
                            clusterization_parameters=_clusterization_parameters)
            for _x, _method, _membership_function, _parameters, _n_last_points_to_use_for_clustering,
                _clusterization_parameters in
            zip(x, methods, membership_functions, parameters, n_last_points_to_use_for_clustering,
                clusterization_parameters)
        ]

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


def cluster_data_1d(x: Union[list, array, Series, DataFrame],
                    method: str = 'gaussian',
                    membership_function: str = 'gaussian',
                    parameters: dict = None,
                    n_last_points_to_use_for_clustering: int = None,
                    normalize: bool = False,
                    clusterization_parameters: dict = None) -> dict:
    """
    Cluster data `x` and calculate membership degrees of `x` to different clusters
    :param x: input data to cluster;
    IF method is one of ('gaussian', 'trapezoidal'): 1D array-like;
    ELIF method is one of ('eTS', 'eClustering'): pandas.DataFrame

    :param method: str, clustering method
    :param membership_function: str, name of the membership function (e.g., 'gaussian')
    :param parameters: dict, parameters of clusters (e.g., number of clusters, centers, variances, etc)
    :param n_last_points_to_use_for_clustering: int, number of last points to use in `x` for clustering
    :param normalize: bool, whether to normalize the resulting membership degrees so that their sum equals 1.
    I.e., whether to divide each membership degree by the sum of membership degrees
    :param clusterization_parameters: dict, parameters of the clustering algorithm
    :return: `dict` {'parameters': estimated parameters of clusters, 'membership': 1D array of membership degrees
    of `x` to clusters[, 'clusterization_parameters': parameters of the clustering method]}
    """

    logger = logging.getLogger('cluster_data_1d')

    slc = slice(-n_last_points_to_use_for_clustering if n_last_points_to_use_for_clustering is not None else None, None)
    x_sliced = x[slc].copy()
    n = len(x_sliced)  # dimension of the space subject to clustering

    logger.debug(f'slc = {slc}')
    logger.debug(f'n = {n}')
    logger.debug(f'x: {x}')
    logger.debug(f'x_sliced: {x_sliced}')

    if method == 'gaussian':
        logger.debug('clustering method is gaussian')

        if parameters is not None:
            logger.debug('parameters is not None')

            centers = parameters['centers']
            variances = parameters['variances']

            centers = [[center] * n for center in centers]
            cov_matrices = [diag([variance] * n, k=0) for variance in variances]

            membership_degrees = calc_gaussian_membership_degrees(x_sliced, centers, cov_matrices)

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

            membership_degrees = calc_trapezoidal_membership_degrees(x_sliced, a, b, c, d)
            result = {'parameters': parameters, 'membership': membership_degrees}
        else:
            raise NotImplementedError('Algorithm for automatic trapezoidal clusterization is not implemented '
                                      '(raised because `parameters` is None)')
    elif method == 'eTS':
        # x is t by (n + 1) or t by n matrix, where t is a current time step, n is a length of the input vector
        # it's t by (n + 1) in case when output (y) is also clustered
        if type(x) is not DataFrame:
            raise ValueError(f'`x` should be a pandas.DataFrame; got `type(x)` = {type(x)}')
        if n_last_points_to_use_for_clustering is not None:
            raise ValueError(f'For clustering method {method} `n_last_points_to_use_for_clustering` should be None; '
                             f'got {n_last_points_to_use_for_clustering}')

        sigma_prev = clusterization_parameters['sigma']
        beta_prev = clusterization_parameters['beta']
        potentials_focal_prev = clusterization_parameters['potentials_focal']
        clusters_variance = clusterization_parameters['variance']
        delta_min = clusterization_parameters['delta_min']
        focals_current = clusterization_parameters['focals']

        t = x.shape[0]
        n = x.shape[1]

        if membership_function == 'gaussian':
            sigma_new, beta_new, focals_new, potentials_focal_new = \
                update_antecedent_part(sigma_prev,
                                       beta_prev,
                                       clusters_variance,
                                       focals_current,
                                       potentials_focal_prev,
                                       x_prev=x.iloc[-2],
                                       x_new=x.iloc[-1],
                                       t=t,
                                       delta_min=delta_min,
                                       )

            parameters_new = {
                'centers': focals_new,
                'variance': clusters_variance
            }

            n_clusters_new = len(focals_new)
            cov_matrices = [diag([clusters_variance] * n, k=0) for _ in range(n_clusters_new)]

            membership_degrees = calc_gaussian_membership_degrees(x.iloc[-1], focals_new, cov_matrices)

            clusterization_parameters_new = clusterization_parameters.copy()
            clusterization_parameters_new.update(
                {
                    'sigma': sigma_new,
                    'beta': beta_new,
                    'focals': focals_new,
                    'potentials_focal': potentials_focal_new
                }
            )

            # resulting dict
            result = {
                'parameters': parameters_new,
                'membership': membership_degrees,
                'clusterization_parameters': clusterization_parameters_new
            }
        else:
            raise ValueError(f'Membership function of form {membership_function} is not supported for '
                             f'clustering method {method}')
    elif method == 'eClustering':
        raise NotImplementedError('eClustering antecedent learning is not implemented')
    else:
        raise ValueError(f'Clustering method name {method} is wrong or method is not implemented; '
                         f'should be one of {CLUSTERING_METHODS}')

    if normalize:
        membership_degrees = membership_degrees / membership_degrees.sum()
        result['membership'] = membership_degrees

    return result
