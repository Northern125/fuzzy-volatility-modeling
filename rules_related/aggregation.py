import logging
from numpy import array


def combine_rules_outputs(outputs, weights):
    logger = logging.getLogger('combine_rules_outputs')

    outputs = array(outputs).copy()
    weights = array(weights).copy()

    weights_sum = weights.sum()
    logger.debug('weights_sum: ' + str(weights_sum))

    if weights_sum == 0:
        raise ZeroDivisionError('weights_sum should not be equal to 0')

    result = (weights * outputs).sum() / weights_sum

    return result
