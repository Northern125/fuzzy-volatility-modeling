from numpy import array
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging
import traceback
from multiprocessing import Pool

from model import FuzzyVolatilityModel


def _try_fitting(_antecedent_params,
                 _n_clusters,

                 alpha_0_ini_1cl,
                 alpha_ini_1cl,
                 beta_ini_1cl,
                 consequent_metaparams,
                 lower_bounds_1cl,
                 upper_bounds_1cl,
                 train,
                 test,
                 clusterization_method,
                 local_method,
                 data_to_cluster_train,
                 data_to_cluster_test,
                 n_last_points_to_use_for_clustering,
                 cluster_sets_conjunction,
                 other_fvm_parameters,
                 q) -> dict:
    logger = logging.getLogger('_try_fitting')

    try:
        # parameters_ini (for LS)
        alpha_0_ini = array([alpha_0_ini_1cl] * _n_clusters)
        alpha_ini = array([alpha_ini_1cl] * _n_clusters).T
        beta_ini = array([beta_ini_1cl] * _n_clusters).T

        parameters_ini = {'alpha_0': alpha_0_ini, 'alpha': alpha_ini, 'beta': beta_ini}

        consequent_metaparams['parameters_ini'] = parameters_ini

        bounds = [[_bounds[0]] * _n_clusters +
                  _bounds[1:1 + q] * _n_clusters +
                  _bounds[1 + q:] * _n_clusters
                  for _bounds in (lower_bounds_1cl, upper_bounds_1cl)]
        bounds = tuple(bounds)

        consequent_metaparams['bounds'] = bounds

        # creating model instance
        fvm = FuzzyVolatilityModel(train,
                                   clusterization_method=clusterization_method,
                                   clusterization_parameters=_antecedent_params,
                                   local_method=local_method,
                                   local_method_parameters=consequent_metaparams,
                                   data_to_cluster=data_to_cluster_train,
                                   n_last_points_to_use_for_clustering=n_last_points_to_use_for_clustering,
                                   cluster_sets_conjunction=cluster_sets_conjunction,
                                   **other_fvm_parameters)

        # clustering, fitting, testing
        fvm.cluster()
        fvm.fit()
        fvm.feed_daily_data(test, data_to_cluster_test)

        # calculating errors
        mse = mean_squared_error(fvm.hist_output, test ** 2, squared=True)
        mape = mean_absolute_percentage_error(fvm.hist_output, test ** 2)

        return {'status': 0,
                'fvm': fvm,
                'mse': mse,
                'mape': mape,
                'exception': None,
                'traceback': None}
    except Exception as e:
        logger.exception(f'Fitting iteration failed: {e}')

        return {'status': -1,
                'fvm': None,
                'mse': None,
                'mape': None,
                'exception': e,
                'traceback': traceback.format_exc()}


def fit_antecedent_params(train,
                          test,
                          consequent_metaparams,
                          consequent_params_ini,
                          antecedent_params_set,
                          clusterization_method='gaussian',
                          local_method='garch',
                          data_to_cluster_train='train',
                          data_to_cluster_test=None,
                          cluster_sets_conjunction=None,
                          n_last_points_to_use_for_clustering=None,
                          n_cluster_sets=None,
                          other_fvm_parameters: dict = None,
                          use_multiprocessing: bool = False,
                          pool_params: dict = None,
                          starmap_params: dict = None) -> list:
    if other_fvm_parameters is None:
        other_fvm_parameters = {}
    if pool_params is None:
        pool_params = {}
    if starmap_params is None:
        starmap_params = {}

    consequent_metaparams = consequent_metaparams.copy()

    # parameters_ini (for LS)
    alpha_0_ini_1cl = consequent_params_ini['alpha_0']
    alpha_ini_1cl = consequent_params_ini['alpha']
    beta_ini_1cl = consequent_params_ini['beta']

    # bounds (for LS)
    lower_bounds_1cl = consequent_metaparams['bounds'][0]
    upper_bounds_1cl = consequent_metaparams['bounds'][1]

    # p & q
    # p = consequent_metaparams['p']  # not needed
    q = consequent_metaparams['q']

    # classes & errors & exceptions
    # fvms = []
    # mses = []
    # mapes = []
    # exceptions = {}
    # tracebacks = {}
    fitting_results = []

    if n_cluster_sets is None:
        # the old logic w/ only one cluster set
        n_clusters = [_cluster_set_params['n_clusters'] for _cluster_set_params in antecedent_params_set]
    else:
        # the new logic w/ possible several cluster sets (e.g., seasonal set & regular volatility set)
        n_clusters = [array([_cluster_set_params['n_clusters'] for _cluster_set_params in _params]).prod()
                      for _params in antecedent_params_set]

    duplicated_params = [
        (
            alpha_0_ini_1cl,
            alpha_ini_1cl,
            beta_ini_1cl,
            consequent_metaparams,
            lower_bounds_1cl,
            upper_bounds_1cl,
            train,
            test,
            clusterization_method,
            local_method,
            data_to_cluster_train,
            data_to_cluster_test,
            n_last_points_to_use_for_clustering,
            cluster_sets_conjunction,
            other_fvm_parameters,
            q
        )
        for _ in range(len(antecedent_params_set))
    ]
    zipped_params = list(zip(antecedent_params_set, n_clusters, duplicated_params))
    unpacked_params = [(_antecedent_params_set, _n_clusters, *_dup) for
                       _antecedent_params_set, _n_clusters, _dup in zipped_params]

    if use_multiprocessing:
        with Pool(**pool_params) as p:
            fitting_results = p.starmap(_try_fitting, unpacked_params, **starmap_params)
    else:
        for _antecedent_params_set, _n_clusters, _other_params in zipped_params:
            fitting_results.append(_try_fitting(_antecedent_params_set, _n_clusters, *_other_params))

    # return {'fvms': fvms, 'mses': mses, 'mapes': mapes, 'exceptions': exceptions, 'tracebacks': tracebacks}

    return fitting_results
