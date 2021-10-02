import logging
from numpy import array


def combine_rules_outputs(outputs, weights):
    logger = logging.getLogger('combine_rules_outputs')

    outputs = array(outputs).copy()
    weights = array(weights).copy()

    weights_sum = weights.sum()
    logger.info(f'weights_sum: {weights_sum}')

    result = (weights * outputs).sum() / weights_sum

    return result
