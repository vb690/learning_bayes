import numpy as np

import pymc3 as pm


def gaussian_estimation(obs_measurement, mu_kwargs, sigma_kwargs):
    """PyMC3 implementation of gaussian estimation.

    Args:
        - obs_measurement: array of float, observed measurement.
        - mu_kwargs: dict, parameters of a normal distrbution.
        - sigma_kwargs: dict, parameters of an halfcauchy distribution.

    Returns:
        - model: a PyMC3 model for stimating mu and sigma parameters of a
            normal distribution.
    """
    with pm.Model() as model:

        mu = pm.Normal(
            'mu',
            **mu_kwargs
        )
        sigma = pm.HalfCauchy(
            'sigma',
            **sigma_kwargs
        )

        measurements = pm.Normal(
            'measurements',
            mu=mu,
            sigma=sigma,
            observed=obs_measurement
        )

    return model


def seven_scientist_estimation(obs_measurements, mu_kwargs, sigma_kwargs):
    """PyMC3 implementation of the seven scientists problem. The general
    idea is the error in measurements can be modelled as varying
    the standard deviation of a normal distribution at the scientist-level.

    Args:
        - obs_measurements: array of float, observed measurements one for
            each scientist.
        - mu_kwargs: dict, parameters of a normal distrbution.
        - sigma_kwargs: dict, parameters of an halfcauchy distribution.

    Returns:
        - model: a PyMC3 model for stimating mu and sigma parameters of a
            normal distribution. Each scientist gets its own sigma.
    """
    with pm.Model() as model:

        n_scientists = len(obs_measurements)
        scientists_indices = pm.Data(
            'scientists_indices',
            np.array([scientist for scientist in range(n_scientists)])
        )

        mu = pm.Normal(
            'mu',
            **mu_kwargs
        )
        sigma = pm.HalfCauchy(
            'sigma',
            shape=(n_scientists,),
            **sigma_kwargs
        )

        measurements = pm.Normal(
            'measurements',
            mu=mu,
            sigma=sigma[scientists_indices],
            observed=obs_measurements
        )

    return model


def repeated_iq_estimation(obs_measurements, id_ind, mu_kwargs,
                           sigma_kwargs):
    """PyMC3 implementation for repeated IQ measurements estimation.

    Args:
        - obs_measurements: array of float, observed measurements can be more
            than one for each individual.
        - id_ind iterable of int, individual id associated to
            each measurement.
        - mu_kwargs: dict, parameters of a normal distrbution.
        - sigma_kwargs: dict, parameters of an halfcauchy distribution.

    Returns:
        - model: a PyMC3 model for stimating mu and sigma parameters of a
            normal distribution. Each id_ind can have repeated measurements.
    """
    with pm.Model() as model:

        id_ind = np.array(id_ind)
        n_ind = len(np.unique(id_ind))

        mu = pm.Normal(
            'mu',
            shape=(n_ind,),
            **mu_kwargs
        )
        sigma = pm.HalfCauchy(
            'sigma',
            **sigma_kwargs
        )

        measurements = pm.Normal(
            'measurements',
            mu=mu,
            sigma=sigma[id_ind],
            observed=obs_measurements
        )

    return model
