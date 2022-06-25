from typing import Union
from numpy import array, vectorize, float64


def calc_trapezoidal_membership_degrees(x: Union[list, array],
                                        a: Union[list, array],
                                        b: Union[list, array],
                                        c: Union[list, array],
                                        d: Union[list, array]) -> array:
    """
    Calculate membership degrees of `x` to trapezoidal clusters given its parameters
    :param x: 1D array-like, data to calculate membership degree for
    :param a: 1D array-like, parameter `a` for each cluster
    :param b: 1D array-like, parameter `b` for each cluster
    :param c: 1D array-like, parameter `c` for each cluster
    :param d: 1D array-like, parameter `d` for each cluster
    :return: 2D `numpy.array`, membership degrees for each cluster (each row corresponds to result of a single cluster)
    """
    result = [calc_trapezoidal_membership_degrees_single_cluster(x, _a, _b, _c, _d)
              for _a, _b, _c, _d in zip(a, b, c, d)]
    result = array(result)

    return result


def calc_trapezoidal_membership_degrees_single_cluster(x: Union[list, array],
                                                       a: float,
                                                       b: float,
                                                       c: float,
                                                       d: float) -> array:
    """
    Calculate membership degrees of `x` to a trapezoidal cluster given its parameters
    :param x: 1D array-like, data to calculate membership degree for
    :param a: float, parameter
    :param b: float, parameter
    :param c: float, parameter
    :param d: float, parameter
    :return: 1D `numpy.array`, membership degrees
    """

    def _calc_md(_x):
        return _calc_trapezoidal_md_scalar(_x, a, b, c, d)

    return vectorize(_calc_md, otypes=[float64])(x)


def _calc_trapezoidal_md_scalar(x, a, b, c, d):
    if a < x < b:
        result = (x - a) / (b - a)
    elif b <= x <= c:
        result = 1
    elif c < x < d:
        result = (d - x) / (d - c)
    else:
        result = 0
    return result
