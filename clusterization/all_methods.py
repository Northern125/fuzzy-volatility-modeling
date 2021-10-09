import logging


def cluster_input(x, method: str = 'hands', parameters: dict = None):
    """

    :param x: input
    :param method: clustering method
    :param parameters parameters of the method
    :return:
    """

    logger = logging.getLogger('cluster_input')

    if method == 'hands':
        logger.debug('clustering method is hands')

        result = parameters
    else:
        logger.warning('clustering method name is wrong or method not yet implemented')
        return

    return result
