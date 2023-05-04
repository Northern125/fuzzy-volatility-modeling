from rules_related import combine_rules_outputs
from numpy import array, apply_along_axis
import logging

PAST_H_TYPE_DEFAULT = 'aggregated'
PAST_H_TYPES = (PAST_H_TYPE_DEFAULT, 'rule-wise')


def calc_ht(alpha_0, alpha, beta, y_squared, h):
    res = alpha_0 + (alpha * y_squared).sum() + (beta * h).sum()
    return res


def calc_fuzzy_ht_aggregated(alpha_0, alpha, beta, y_squared, h,
                             weights: array,
                             ) -> array:
    """

    :param alpha_0:
    :param alpha:
    :param beta:
    :param y_squared:
    :param h: 1D or 2D array-like, depending on `past_h_type`. The past conditional variance values `h_{t-j}`
    :param weights:
    aggregated h values. If 'rule-wise': `h` should be 2 dimensional and contain rule-wise `h_{t-j}^(k)` values
    :return: 1D array. Output of each cluster
    """
    h = array(h).copy()

    n_clusters: int = len(weights)

    outputs = [
        calc_ht(alpha_0[j], alpha[:, j], beta[:, j], y_squared, h)
        for j in range(n_clusters)
    ]
    outputs = array(outputs).copy()

    output = combine_rules_outputs(outputs, weights).copy()

    return output


def calc_fuzzy_ht_rule_wise(alpha_0, alpha, beta, y_squared, h,
                            n_clusters: int):
    """

    :param alpha_0:
    :param alpha:
    :param beta:
    :param y_squared:
    :param h:
    :param n_clusters:
    :return:
    """
    h = array(h).copy()

    outputs = [
        calc_ht(alpha_0[j], alpha[:, j], beta[:, j], y_squared, h[:, j])
        for j in range(n_clusters)
    ]
    outputs = array(outputs).copy()

    return outputs


def _calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                   calc_ht_function: callable = calc_ht,
                   **kwargs) -> array:
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

    for i in range(starting_index, y_len + 1):
        y_slc = slice(i - q, i)
        h_slc = slice(i - p, i)
        h_t = calc_ht_function(alpha_0, alpha, beta, y_squared[y_slc], h[h_slc], **kwargs)
        h.append(h_t)
        logger.debug(f'New iteration; i = {i}: h_t = {h_t}, y_slc = {y_slc}, h_slc = {h_slc}, '
                     f'h[h_slc] = {h[h_slc]}, y_squared[y_slc] =\n{y_squared[y_slc]}')

    h = array(h).copy()

    return h


def calc_cond_var_vanilla(alpha_0, alpha, beta, y_squared, first_h) -> array:
    return _calc_cond_var(alpha_0, alpha, beta, y_squared, first_h,
                          calc_ht_function=calc_ht)


def calc_cond_var_fuzzy(alpha_0, alpha, beta, y_squared, first_h, weights,
                        return_fuzzy: bool = False,
                        past_h_type: str = PAST_H_TYPE_DEFAULT) -> array:
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
