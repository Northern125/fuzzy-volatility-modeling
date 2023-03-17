import logging
from numpy import array, concatenate, array_str, inf

logger = logging.getLogger('ets_antecedent')


def initialize_default_parameters(first_data_point: array):
    logger.info('initializing default params')

    focal_points = array([first_data_point])
    potentials_focal = array([1])

    return focal_points, potentials_focal


def update_antecedent_part(sigma_prev: float,
                           beta_prev: float,
                           clusters_variance: float,
                           focals_current: array,
                           potentials_focal_prev: array,
                           x_prev: array,
                           x_new: array,
                           t: int,
                           delta_min: float,
                           ) -> tuple:
    """
    Update antecedent parameters via the eTS (evolving Takagi - Sugeno [Angelov, Filev, 2004]) method

    :param sigma_prev:
    :param beta_prev:
    :param clusters_variance: float
    :param focals_current: 2D numpy.array of size (R by n) = (`len(potentials_focal_prev)` by `len(x_prev)`)
    :param potentials_focal_prev:
    :param x_prev:
    :param x_new:
    :param t:
    :param delta_min:

    :return:
    """
    # x is t by (n + 1) or t by n matrix, where t is a current time step, n is a length of the input vector
    # it's t by (n + 1) in case when output (y) is also clustered

    logger.info(f'starting antecedent update; x_prev = {x_prev}; x_new = {x_new}')

    theta_current = (x_new ** 2).sum()
    sigma_current = sigma_prev + (x_prev ** 2).sum()
    beta_current = beta_prev + x_prev
    nu_current = (x_new * beta_current).sum()

    logger.debug(f'theta_current = {theta_current}; sigma_current = {sigma_current}; '
                 f'beta_current = {beta_current}; nu_current = {nu_current}')

    potential_new_point = (t - 1) / ((t - 1) * (theta_current + 1) + sigma_current - 2 * nu_current)
    logger.debug(f'potential_new_point = {potential_new_point}')
    potentials_focal_new = ((t - 1) * potentials_focal_prev) / \
                           (t - 2 + potentials_focal_prev +
                            potentials_focal_prev * ((x_new - x_prev) ** 2).sum())
    logger.debug(f'potentials_focal_new = {potentials_focal_new}')

    current_potentials_max = potentials_focal_new.max()
    max_potential_idx = potentials_focal_new.argmax()

    logger.debug(f'potentials_focal_new[{max_potential_idx}] = {current_potentials_max} is a maximum')

    focals_new = focals_current.copy()

    if potential_new_point > current_potentials_max:
        logger.info('potential of a new point is greater than those of all current focals')
        if potential_new_point / current_potentials_max - delta_min / clusters_variance >= 1:
            logger.info('new point is close to an existing focal point')

            focals_new[max_potential_idx] = x_new
            potentials_focal_new[max_potential_idx] = potential_new_point
        else:
            logger.info('new point forms a new cluster')

            # focals_new = list(focals_current).copy()
            # focals_new.append(x_new)
            # focals_new = array(focals_new).copy()
            focals_new = concatenate((focals_current, [x_new]), axis=0).copy()
            logger.debug(f'type(focals_new) = {type(focals_new)}')
            focals_new_str = array_str(focals_new, max_line_width=inf).replace('\n', '')
            logger.debug(f"""focals_new = {focals_new_str}""")

            # potentials_focal_new = list(potentials_focal_new).copy()
            # potentials_focal_new.append(potential_new_point)
            # potentials_focal_new = array(potentials_focal_new).copy()
            potentials_focal_new = concatenate((potentials_focal_new, [potential_new_point]), axis=0).copy()
            logger.debug(f'potentials_focal_new = {potentials_focal_new}')
            logger.debug(f'type(potentials_focal_new) = {type(potentials_focal_new)}')

    return sigma_current, beta_current, focals_new, potentials_focal_new


def calc_sigma(x: array) -> float:
    return x[:-1, :].sum()


def calc_beta(x: array) -> array:
    return x[:-1, :].sum(axis=0)
