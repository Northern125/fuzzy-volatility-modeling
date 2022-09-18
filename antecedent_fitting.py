import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging
import traceback

from model import FuzzyVolatilityModel


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
                          use_multiprocessing: bool = False):
    logger = logging.getLogger('fit_antecedent_params')

    if other_fvm_parameters is None:
        other_fvm_parameters = {}

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
    fvms = []
    mses = []
    mapes = []
    exceptions = {}
    tracebacks = {}

    if n_cluster_sets is None:
        # the old logic w/ only one cluster set
        n_clusters = [_cluster_set_params['n_clusters'] for _cluster_set_params in antecedent_params_set]
    else:
        # the new logic w/ possible several cluster sets (e.g., seasonal set & regular volatility set)
        n_clusters = [np.array([_cluster_set_params['n_clusters'] for _cluster_set_params in _params]).prod()
                      for _params in antecedent_params_set]

    def _try_fitting(_antecedent_params, _n_clusters):
        try:
            # parameters_ini (for LS)
            alpha_0_ini = np.array([alpha_0_ini_1cl] * _n_clusters)
            alpha_ini = np.array([alpha_ini_1cl] * _n_clusters).T
            beta_ini = np.array([beta_ini_1cl] * _n_clusters).T

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

    if use_multiprocessing:
        raise NotImplementedError('Multiprocessing use not yet implemented')
    else:
        for i, (antecedent_params, _n_clusters) in enumerate(zip(antecedent_params_set, n_clusters)):
            logger.debug(f'Starting iteration #{i}')

            fitting_result = _try_fitting(antecedent_params, _n_clusters)

            fvms.append(fitting_result['fvm'])
            mses.append(fitting_result['mse'])
            mapes.append(fitting_result['mape'])
            if fitting_result['status'] == -1:
                e = fitting_result['exception']
                logger.exception(f'Iteration #{i} failed: {e}')
                exceptions[i] = e
                tracebacks[i] = fitting_result['traceback']
            else:
                logger.debug(f'Iteration #{i} completed successfully')

    return {'fvms': fvms, 'mses': mses, 'mapes': mapes, 'exceptions': exceptions, 'tracebacks': tracebacks}
