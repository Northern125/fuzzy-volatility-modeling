import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging

from model import FuzzyVolatilityModel


def fit_antecedent_params(train, test,
                          consequent_metaparams,
                          consequent_params_ini,
                          antecedent_params_set,
                          clusterization_method='gaussian',
                          local_method='garch',
                          data_to_cluster_train='train', data_to_cluster_test=None):
    logger = logging.getLogger('fit_antecedent_params')

    consequent_metaparams = consequent_metaparams

    # parameters_ini (for LS)
    alpha_0_ini = consequent_params_ini['alpha_0']
    alpha_ini = consequent_params_ini['alpha']
    beta_ini = consequent_params_ini['beta']

    # p & q
    p = consequent_metaparams['p']
    q = consequent_metaparams['q']

    #
    fvms = []
    mses = []
    mapes = []

    i = 0
    for antecedent_params in antecedent_params_set:
        logger.info(f'Starting iteration #{i}')

        n_clusters = antecedent_params['n_clusters']

        # parameters_ini (for LS)
        alpha_0_ini = np.array([alpha_0_ini] * n_clusters)
        alpha_ini = np.array([alpha_ini] * n_clusters).T
        beta_ini = np.array([beta_ini] * n_clusters).T

        parameters_ini = {'alpha_0': alpha_0_ini, 'alpha': alpha_ini, 'beta': beta_ini}

        consequent_metaparams['parameters_ini'] = parameters_ini

        # bounds (for LS)
        lower_bounds = [0] * ((1 + p + q) * n_clusters)
        upper_bounds = [+np.inf] * n_clusters + [1] * ((p + q) * n_clusters)
        bounds = (lower_bounds, upper_bounds)

        consequent_metaparams['bounds'] = bounds

        # creating model instance
        fvm = FuzzyVolatilityModel(train,
                                   clusterization_method=clusterization_method,
                                   clusterization_parameters=antecedent_params,
                                   local_method=local_method,
                                   local_method_parameters=consequent_metaparams,
                                   data_to_cluster=data_to_cluster_train)

        # fitting
        fvm.fit()

        # testing
        fvm.feed_daily_data(test, data_to_cluster_test)
        fvms.append(fvm)

        # calculating errors
        mse = mean_squared_error(fvm.hist_output, test ** 2, squared=True)
        mape = mean_absolute_percentage_error(fvm.hist_output, test ** 2)
        mses.append(mse)
        mapes.append(mape)

        logger.info(f'Iteration #{i} ended')
        i += 1

    return {'fvms': fvms, 'mses': mses, 'mapes': mapes}
