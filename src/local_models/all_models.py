import logging
from arch import arch_model
from numpy import array


def apply_local_model(input_data,
                      method: str = 'garch',
                      parameters: dict = None,
                      forecast_horizon=1):
    logger = logging.getLogger('apply_local_model')

    input_data = array(input_data).copy()

    if method == 'garch':
        logger.debug('Method is GARCH')

        p = parameters['p']
        q = parameters['q']
        mean = parameters['mean']
        dist = parameters['dist']

        model = arch_model(input_data, mean=mean, vol='GARCH', p=p, q=q, dist=dist)
        fitted = model.fit()

        estimated_params = fitted.params

        forecast = fitted.forecast(reindex=False, horizon=forecast_horizon).variance.values[0]

        return {'parameters': estimated_params, 'forecast': forecast}
    else:
        logger.warning('Method name is wrong or method not yet implemented. Exiting')
        return

    pass
