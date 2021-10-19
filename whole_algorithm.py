import logging
from numpy import array

from local_models import apply_local_model
from rules_related import combine_rules_outputs
from clusterization import cluster_data


def train_model(input_data,
                clusterization_method: str = 'gaussian',
                clusterization_parameters: dict = None,
                local_method: str = 'garch',
                local_method_parameters: dict = None):
    logger = logging.getLogger('train_model')

    # clusterization
    logger.debug('Starting clusterization')

    clusterization_result = cluster_data(input_data,
                                         method=clusterization_method,
                                         parameters=clusterization_parameters)

    clusters_parameters = clusterization_result['parameters']
    n_clusters = clusters_parameters['n_clusters']

    membership_degrees = clusterization_result['membership']

    logger.debug(f'Clusterization completed\n'
                 f'Estimated parameters: {clusters_parameters}\n'
                 f'Membership degrees:\n{membership_degrees}')

    # running local models for each rule
    logger.debug('Starting to run local model for each rule')

    rules_outputs = []
    for i in range(n_clusters):
        rule_output = apply_local_model(input_data,
                                        method=local_method,
                                        parameters=local_method_parameters,
                                        forecast_horizon=1)['forecast']
        rules_outputs.append(rule_output)

    rules_outputs = array(rules_outputs)

    logger.debug(f'Local model runs for each rule are completed. rules_outputs: {rules_outputs}')

    # aggregating rules outputs to a single output
    logger.debug('Starting to aggregate all rules outputs to a single one')

    combined_output = combine_rules_outputs(rules_outputs, membership_degrees)

    return combined_output


def testing_with_retraining(train_data, test_data=None, n_test=None):
    logger = logging.getLogger('testing_with_retraining')

    train_data = array(train_data).copy()

    if test_data is not None:
        test_data = array(test_data).copy()
    else:
        if n_test is not None:
            test_data = train_data[-n_test:].copy()
            train_data = train_data[:-n_test].copy()
        else:
            logger.exception('Either test_data or n_test should be not None; exiting')
            return

    logger.debug(f'test_data: {test_data}\n'
                 f'train_data: {train_data}')

    pass
