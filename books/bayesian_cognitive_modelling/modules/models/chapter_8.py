import pymc as pm


def estimate_mean_difference_one_sample(obs_measures, delta_cauchy_kwargs,
                                        sigma_cauchy_kwargs, side=None):
    """PyMC3 implementation of one sample mean comparison

    Args:
        - obs_measures: numpy array, observed measurements.
        - delta_cauchy_kwargs: dict, parameters of a Cauchy distribution, it
            is half Cauchy if one_side = True.
        - sigma_cauchy_kwargs: dict, parameters of an half Cauchy distribution.
        - one_side: bool, whther the test is one or two sided.

    Returns:
        - model
    """
    with pm.Model() as model:

        if side is not None:
            side_multiplier = -1 if side == 'left' else 1
            delta = pm.Deterministic(
                'effect_size',
                side_multiplier * pm.HalfCauchy(
                    'effect_size_cauchy',
                    **delta_cauchy_kwargs
                )
            )
            if side == 'left':
                delta = delta * - 1
        else:
            delta = pm.Cauchy(
                'effect_size',
                **delta_cauchy_kwargs
            )

        sigma = pm.HalfCauchy(
            'standard_deviation',
            **sigma_cauchy_kwargs
        )

        mu = pm.Deterministic(
            'mu',
            delta*sigma
        )

        observed_measures = pm.Normal(
            'observed_measures',
            mu=mu,
            sigma=sigma,
            observed=obs_measures
        )
    return model


def estimate_mean_difference_two_samples(obs_measure, delta_cauchy_kwargs,
                                         sigma_cauchy_kwargs,
                                         mu_normal_kwargs):
    """
    """
    group_1 = obs_measures[:, 0]
    group_2 = obs_measures[:, 1]

    with pm.Model() as model:

        delta = pm.Cauchy(
            'effect_size',
            **delta_cauchy_kwargs
        )
        sigma = pm.HalfCauchy(
            'standard_deviation',
            **sigma_cauchy_kwargs
        )

        alpha = pm.Deterministic(
            'mean_difference',
            delta * sigma
        )

        mean = pm.Normal(
            'mu',
            **mu_normal_kwargs
        )

        group_1_obs = pm.Normal(
            'group_1',
            mu=mean+alpha,
            sigma=sigma,
            observed=group_1
        )
        group_2_obs = pm.Normal(
            'group_1',
            mu=mean-alpha,
            sigma=sigma,
            observed=group_2
        )

    return model
