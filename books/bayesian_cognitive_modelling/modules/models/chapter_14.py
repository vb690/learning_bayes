import numpy as np

from scipy.stats import norm

import pymc3 as pm


def multinomial_processing_tree(obs_responses, c_beta_kwargs,
                                r_beta_kwargs, u_beta_kwargs):
    """
    Responses are

    C1: both words recalled consecutively.
    C2: both words recalled not consecutively.
    C3: only one word is recalled.
    C4: neither words are recalled.
    """
    with pm.model() as model:

        c = pm.Beta(
            'c',
            **c_beta_kwargs
        )
        r = pm.Beta(
            'r',
            **r_beta_kwargs
        )
        u = pm.Beta(
            'u',
            **u_beta_kwargs
        )

        theta_c1 = pm.Deterministic(
            'c1',
            c*r
        )
        theta_c2 = pm.Deterministic(
            'c2',
            (1-c)*(u**2)
        )
        theta_c3 = pm.Deterministic(
            'c3',
            (2*u)*(1-c)*(1-u)
        )
        theta_c4 = pm.Deterministic(
            'c4',
            c*(1-r)+(1-c)*((1-u)**2)
        )

        observed_responses = pm.Multinomial(
            'observed_responses',
            p=[theta_c1, theta_c2, theta_c3, theta_c4],
            n=obs_responses
        )

    return model


def latent_MPT(id_ind, obs_responses, c_sigma_kwargs, r_sigma_kwargs,
               u_sigma_kwargs, cholesky_kwargs):
    """
    Responses are

    C1: both words recalled consecutively.
    C2: both words recalled not consecutively.
    C3: only one word is recalled.
    C4: neither words are recalled.
    """
    with pm.Model() as model:

        id_ind_data = pm.Data(
            'id_ind',
            id_ind
        )

        c_mu = pm.Normal(
            'c_mu',
            0,
            1
        )
        r_mu = pm.Normal(
            'r_mu',
            0,
            1
        )
        u_mu = pm.Normal(
            'u_mu',
            0,
            1
        )

        c_sigma = pm.HalfCauchy(
            'c_sigma',
            **c_sigma_kwargs
        )
        r_sigma = pm.HalfCauchy(
            'r_sigma',
            **r_sigma_kwargs
        )
        u_sigma = pm.HalfCauchy(
            'u_sigma',
            **u_sigma_kwargs
        )

        chol, corr, stds = pm.LKJCholeskyCov(
            'cholesky',
            n=3,
            compute_corr=True,
            **cholesky_kwargs
        )

        deltas = pm.MvNormal(
            'deltas',
            mu=[0, 0, 0],
            chol=chol,
            shape=(len(id_ind), 3)
        )

        c = pm.Deterministic(
            'c',
            norm.cdf(
                c_mu + (c_sigma*deltas[id_ind_data, 0])
            )
        )
        r = pm.Deterministic(
            'r',
            norm.cdf(
                r_mu + (r_sigma*deltas[id_ind_data, 1])
            )
        )
        u = pm.Deterministic(
            'u',
            norm.cdf(
                u_mu + (u_sigma*deltas[id_ind_data, 2])
            )
        )

        theta_c1 = pm.Deterministic(
            'c1',
            c*r
        )
        theta_c2 = pm.Deterministic(
            'c2',
            (1-c)*(u**2)
        )
        theta_c3 = pm.Deterministic(
            'c3',
            (2*u)*(1-c)*(1-u)
        )
        theta_c4 = pm.Deterministic(
            'c4',
            c*(1-r)+(1-c)*((1-u)**2)
        )

        observed_responses = pm.Multinomial(
            'observed_responses',
            p=[theta_c1, theta_c2, theta_c3, theta_c4],
            n=obs_responses
        )

    return model
