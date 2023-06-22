from typing import Union

from src.rules_related import combine_rules_outputs
from numpy import array, apply_along_axis
import logging

PAST_H_TYPE_DEFAULT = 'aggregated'
PAST_H_TYPES = (PAST_H_TYPE_DEFAULT, 'rule-wise')


def calc_ht(alpha_0: float,
            alpha: array,
            beta: array,
            y_squared: array,
            h: array) -> float:
    """
    Calculate vanilla (non-fuzzy) heteroskedasticity value at step t given GARCH parameters & y^2 and h lagged values.
    `y_squared` and `h` are passed in the reverse order!

    :param alpha_0: float. GARCH parameter
    :param alpha: 1D numpy.array of length `q`. GARCH parameter
    :param beta: 1D numpy.array of length `p`. GARCH parameter
    :param y_squared: 1D numpy.array of shape (`len(alpha)`, ). Last `q` lagged y^2 values in reverse order
    :param h: 1D numpy.array if shape (`len(beta)`, ). Last `p` lagged conditional variance values in reverse order
    :return: float. Heteroskedasticity `h_t` value at current time step
    """
    res = alpha_0 + (alpha * y_squared).sum() + (beta * h).sum()
    return res


def calc_fuzzy_ht_aggregated(alpha_0: array,
                             alpha: array,
                             beta: array,
                             y_squared: array,
                             h: array,
                             weights: array,
                             ) -> float:
    """
    Calculate fuzzy heteroskedasticity aggregated value at step t given GARCH parameters & y^2 and h lagged values.
    Aggregated means `h` lags used in the GARCH formula are aggregated, not rule-wise.
    `y_squared` and `h` are passed in the reverse order!

    :param alpha_0: 1D numpy.array of shape `(n_clusters, )`. GARCH parameter for each cluster
    :param alpha: 2D numpy.array of shape `(q, n_clusters)`. GARCH parameter for each cluster
    :param beta: 2D numpy.array of shape `(p, n_clusters)`. GARCH parameter for each cluster
    :param y_squared: 1D numpy.array of shape (`len(alpha)`, ). Last `q` lagged y^2 values in reverse order
    :param h: 1D numpy.array of shape (`len(beta)`, ). The AGGREGATED `p` past conditional variance values `h_{t-j}` in
    reverse order
    :param weights: 1D numpy.array of length `n_clusters`. Weight of each cluster (for output aggregation).
    `n_clusters` is inferred from this parameter
    :return: float. Aggregated conditional variance `h_t` value at current time step
    """
    n_clusters: int = len(weights)

    outputs = [
        calc_ht(alpha_0[j], alpha[:, j], beta[:, j], y_squared, h)
        for j in range(n_clusters)
    ]
    outputs = array(outputs).copy()

    output = combine_rules_outputs(outputs, weights).copy()

    return output


def calc_fuzzy_ht_rule_wise(alpha_0: array,
                            alpha: array,
                            beta: array,
                            y_squared: array,
                            h: array,
                            n_clusters: int
                            ) -> array:
    """
    Calculate fuzzy heteroskedasticity aggregated value at step t given GARCH parameters & y^2 and h lagged values.
    Rule-wise `h` lags are used in the GARCH formula (own `h` for each fuzzy rule).
    `y_squared` and `h` are passed in the reverse order!

    :param alpha_0: 1D numpy.array of shape `(n_clusters, )`. GARCH parameter for each cluster
    :param alpha: 2D numpy.array of shape `(q, n_clusters)`. GARCH parameter for each cluster
    :param beta: 2D numpy.array of shape `(p, n_clusters)`. GARCH parameter for each cluster
    :param y_squared: 1D numpy.array of length `q`. Last `q` lagged y^2 values in reverse order
    :param h: 2D numpy.array of shape `(p, n_clusters)`. Last `p` conditional variance values `h_{t-j}^(k)` in reverse
    order FOR EACH CLUSTER
    :param n_clusters: int. Number of clusters
    :return: 1D numpy.array of length `n_clusters`. Conditional variance `h_t^(k)` value at current time step for
    each cluster
    """
    outputs = [
        calc_ht(alpha_0[j], alpha[:, j], beta[:, j], y_squared, h[:, j])
        for j in range(n_clusters)
    ]
    outputs = array(outputs).copy()

    return outputs


def _calc_cond_var(alpha_0: Union[float, array],
                   alpha: array,
                   beta: array,
                   y_squared,
                   first_h,
                   calc_ht_function: callable = calc_ht,
                   **kwargs) -> array:
    """
    Calculate conditional variance recursively from the beginning of the `y_squared` series. Depending on the
    `calc_ht_function` argument, the calculation method can be either a vanilla GARCH, a fuzzy GARCH w/ aggregated `h`
    or a fuzzy GARCH w/ rule-wise `h`

    :param alpha_0: float for vanilla GARCH, 1D numpy.array of len `n_clusters` for fuzzy GARCH
    :param alpha: 1D numpy.array of len `q` for vanilla GARCH, 2D numpy.array of shape `(q, n_clusters)` for fuzzy GARCH
    :param beta: 1D numpy.array of len `q` for vanilla GARCH, 2D numpy.array of shape `(q, n_clusters)` for fuzzy GARCH
    :param y_squared: 1D numpy.array of any length
    :param first_h: 1D numpy.array of len `p` for vanilla GARCH & fuzzy GARCH w/ aggregated `h`,
    2D numpy.array of shape `(p, n_clusters)` for fuzzy GARCH w/ rule-wise (fuzzy) `h`.
    The first `p` values of the conditional variance
    :param calc_ht_function: callable, one of (calc_ht, calc_fuzzy_ht_aggregated, calc_fuzzy_ht_rule_wise)
    :param kwargs: kwargs for `calc_ht_function`
    :return: 1D numpy.array of length `len(y_squared) + 1` for vanilla GARCH & fuzzy GARCH w/ aggregated `h`
    and 2D numpy array of shape `(len(y_squared) + 1, n_clusters)` for fuzzy GARCH w/ rule-wise (fuzzy) `h`.
    The recursively calculated conditional variance values from time 0 up to time step t + 1
    """
    logger = logging.getLogger('_calc_cond_var')

    q = len(alpha)
    p = len(beta)

    starting_index = max(p, q)
    logger.debug(f'starting_index = {starting_index}')

    if len(first_h) < starting_index:
        raise Exception(f'Not enough first elements of h are given. p = {p}, q = {q}, max(p, q) = {starting_index}. '
                        f'Therefore, we need first {starting_index} elements of h known. '
                        f'However only {len(first_h)} are given: given first_h = {first_h}')

    h = list(first_h)

    y_len = len(y_squared)
    logger.debug(f'y_len = {y_len}')

    for t in range(starting_index, y_len + 1):
        y_slc = slice(t - q, t)
        h_slc = slice(t - p, t)
        _y_squared = array(list(reversed(y_squared[y_slc])))
        _h = array(list(reversed(h[h_slc])))
        h_t = calc_ht_function(alpha_0, alpha, beta, _y_squared, _h, **kwargs)
        h.append(h_t)
        logger.debug(f'New iteration; t = {t}: h_t = {h_t}, y_slc = {y_slc}, h_slc = {h_slc}, '
                     f'h[h_slc] = {h[h_slc]}, y_squared[y_slc] =\n{y_squared[y_slc]}')

    h = array(h).copy()

    return h


def calc_cond_var_vanilla(alpha_0: float,
                          alpha: array,
                          beta: array,
                          y_squared: array,
                          first_h: array
                          ) -> array:
    """
    A wrapper function for conditional variance calculation for vanilla GARCH.
    Calculate conditional variance recursively from the beginning of the `y_squared` series.

    :param alpha_0: float. GARCH parameter
    :param alpha: 1D numpy.array of length `q`. GARCH parameter
    :param beta: 1D numpy.array of length `p`. GARCH parameter
    :param y_squared: 1D numpy.array of any length
    :param first_h: 1D numpy.array of len `p` for vanilla GARCH. The first `p` values of the conditional variance
    :return: 1D numpy.array of length `len(y_squared) + 1`.
    The recursively calculated conditional variance values from time 0 up to time step t + 1
    """
    return _calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                          calc_ht_function=calc_ht)


def calc_cond_var_fuzzy(alpha_0: array,
                        alpha: array,
                        beta: array,
                        y_squared: array,
                        first_h: array,
                        weights: array,
                        return_fuzzy: bool = False,
                        past_h_type: str = PAST_H_TYPE_DEFAULT
                        ) -> Union[tuple[array, array], array]:
    """
    A wrapper function for conditional variance calculation for fuzzy GARCH.
    Calculate conditional variance recursively from the beginning of the `y_squared` series

    :param alpha_0: 1D numpy.array of shape `(n_clusters, )`. GARCH parameter for each cluster
    :param alpha: 2D numpy.array of shape `(q, n_clusters)`. GARCH parameter for each cluster
    :param beta: 2D numpy.array of shape `(p, n_clusters)`. GARCH parameter for each cluster
    :param y_squared: 1D numpy.array of any length
    :param first_h: The first `p` values of the conditional variance `h`.
    If `past_h_type == rule-wise`: 2D numpy.array of shape `(p, n_clusters)`; contains rule-wise values.
    If `past_h_type == aggregated`: 1D numpy.array of len `p`; contains aggregated values
    :param weights: 1D numpy.array of length `n_clusters`. Weight of each cluster (for output aggregation)
    :param return_fuzzy: bool. If True, the function also returns fuzzy conditional variance
    :param past_h_type: str. Type of past conditional variance ('rule-wise' AKA fuzzy or 'aggregated')
    :return: If `return_fuzzy == True`: `tuple(fuzzy_h, aggr_h)`.
    If `return_fuzzy == False`: `aggr_h`.
    `aggr_h` is a 1D numpy.array of length `len(y_squared) + 1`, aggregated conditional variance.
    `fuzzy_h` is a 2D numpy.array of shape `(len(y_squared) + 1, n_clusters)`, fuzzy conditional variance.
    The recursively calculated conditional variance values from time 0 up to time step t + 1
    """
    logger = logging.getLogger('calc_cond_var_fuzzy')

    n_clusters: int = weights.shape[0]

    if past_h_type == 'aggregated':
        fuzzy_cond_var = _calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                                        weights=weights,
                                        calc_ht_function=calc_fuzzy_ht_aggregated)
        cond_var_combined = fuzzy_cond_var.copy()
    elif past_h_type == 'rule-wise':
        fuzzy_cond_var = _calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                                        n_clusters=n_clusters,
                                        calc_ht_function=calc_fuzzy_ht_rule_wise)
        cond_var_combined = apply_along_axis(combine_rules_outputs,
                                             axis=1,
                                             arr=fuzzy_cond_var,
                                             weights=weights)
    else:
        raise ValueError(f'`past_h_type` should be one of {PAST_H_TYPES}; got {past_h_type}')

    logger.debug(f'fuzzy_cond_var = {fuzzy_cond_var.tolist()}')

    if return_fuzzy:
        return fuzzy_cond_var, cond_var_combined
    else:
        return cond_var_combined
