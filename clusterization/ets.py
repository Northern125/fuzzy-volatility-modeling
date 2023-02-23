from numpy import array


def initialize_default_parameters(first_data_point: array):
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

    theta_current = (x_new ** 2).sum()
    sigma_current = sigma_prev + (x_prev ** 2).sum()
    beta_current = beta_prev + x_prev
    nu_current = (x_new * beta_current).sum()

    potential_new_point = (t - 1) / ((t - 1) * (theta_current + 1) + sigma_current - 2 * nu_current)
    potentials_focal_new = ((t - 1) * potentials_focal_prev) / \
                           (t - 2 + potentials_focal_prev +
                            potentials_focal_prev * ((x_new - x_prev) ** 2).sum())

    current_potentials_max = potentials_focal_new.max()
    max_potential_idx = potentials_focal_new.argmax()

    focals_new = focals_current.copy()

    if potential_new_point > current_potentials_max:
        if potential_new_point / current_potentials_max - delta_min / clusters_variance >= 1:
            focals_new[max_potential_idx] = x_new
            potentials_focal_new[max_potential_idx] = potential_new_point
        else:
            focals_new = list(focals_current).copy()
            focals_new.append(x_new)
            focals_new = array(focals_new)

            potentials_focal_new = list(potentials_focal_new).copy()
            potentials_focal_new.append(potential_new_point)
            potentials_focal_new = array(potentials_focal_new)

    return sigma_current, beta_current, focals_new, potentials_focal_new
