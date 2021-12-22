import numpy as np

import pymc3 as pm


def multinomial_processing_tree(obs_responses, c_beta_kwargs, r_beta_kwargs,
                                u_beta_kwargs):
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
