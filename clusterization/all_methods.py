import logging

from clusterization.membership_functions import calc_membership_degree


def cluster_data(x, method: str = 'gaussian', parameters: dict = None):
    """

    :param x: input
    :param method: clustering method
    :param parameters: parameters of the clusterization algorithm
    :return:
    """

    logger = logging.getLogger('cluster_input')

    if method == 'gaussian':
        logger.debug('clustering method is gaussian')

        if parameters is not None:
            logger.debug('parameters is not None')

            # TODO this is wrong - this is a logic for just 1 cluster; need to write logic for all clusters
            membership_degrees = calc_membership_degree(x, membership_function_type=method, parameters=parameters)
            result = {'parameters': parameters, 'membership': membership_degrees}
        else:
            logger.debug('parameters is None')
            logger.warning('Algorithm for automatic Gaussian clusterization is not yet implemented. Exiting')
            return
    else:
        logger.warning('Clustering method name is wrong or method not yet implemented')
        return

    return result
