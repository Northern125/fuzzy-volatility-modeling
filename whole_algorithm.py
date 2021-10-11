import logging
from numpy import array

from local_models import apply_local_model
from rules_related import combine_rules_outputs
from clusterization import cluster_data


def train(input_data,
          actual_output,
          clusterization_method: str = 'gaussian',
          clusterization_parameters: dict = None,
          local_method='garch',
          local_method_parameters: dict = None):
    logger = logging.getLogger('train')

    # clusterization
    logger.debug('Starting clusterization')

    clusterization_result = cluster_data(input_data,
                                         method=clusterization_method,
                                         parameters=clusterization_parameters)
    clusters_parameters = clusterization_result['parameters']
    n_clusters = clusters_parameters['n_clusters']
    membership_degrees = clusters_parameters['membership']

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
                                        forecast_horizon=1)
        rules_outputs.append(rule_output)

    rules_outputs = array(rules_outputs)

    logger.debug(f'Local model runs for each rule are completed. rules_outputs: {rules_outputs}')

    # aggregating rules outputs to a single output
    logger.debug('Starting to aggregate all rules outputs to a single one')

    combined_output = combine_rules_outputs(rules_outputs, )

    pass
