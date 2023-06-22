from numpy import array
from arch import arch_model


def calculate_retraining_garch_forecasts(train, test,
                                         mean='Constant', vol='GARCH', p=1, q=1, dist='normal',
                                         n_points_fitting: int = None):
    _fitting_slice = slice(-n_points_fitting if n_points_fitting is not None else None, None)

    train = train.copy()
    test = test.copy()

    garch = arch_model(train[_fitting_slice],
                       mean=mean,
                       vol=vol,
                       p=p,
                       q=q,
                       dist=dist)
    garch_fitted = garch.fit()
    one_step_forecast = garch_fitted.forecast(horizon=1, reindex=False).variance.iloc[0].values[0]

    forecast = [one_step_forecast]

    for date in test.index[:-1]:
        observation = test.loc[date]
        train.loc[date] = observation

        # creating garch model instance
        garch_mean = 'Zero'
        garch_dist = 'normal'

        garch = arch_model(train[_fitting_slice],
                           mean=mean,
                           vol=vol,
                           p=p,
                           q=q,
                           dist=dist)

        # fitting
        garch_fitted = garch.fit()

        #
        one_step_forecast = garch_fitted.forecast(horizon=1, reindex=False).variance.iloc[0].values[0]
        forecast.append(one_step_forecast)

    return array(forecast)
