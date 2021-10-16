from numpy import log


def calc_log_returns(series):
    series = series.copy()

    result = series.rolling(2).apply(lambda values: log(values[1] / values[0])).copy()

    return result
