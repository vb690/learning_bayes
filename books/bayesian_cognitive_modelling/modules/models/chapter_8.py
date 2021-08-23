import numpy as np

import pymc3 as pm


def estimate_mean_difference_one_sample(obs_measures, delta_cauchy_kwargs,
                                        sigma_cauchy_kwargs, one_side=False):
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

        if one_side:
            delta = pm.HalfCauchy(
                'effect_size',
                **delta_cauchy_kwargs
            )
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
