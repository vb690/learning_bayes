import numpy as np

import pymc3 as pm


def duration_discrimination(id_ind, n_trials, times, obs_responses,
                            alpha_sigma_kwargs, alpha_mu_kwargs,
                            beta_sigma_kwargs, beta_mu_kwargs,
                            ):
    """
    """
    with pm.Model() as model:

        id_ind_data = pm.Data(
            'id_ind',
            id_ind
        )
        n_trials_data = pm.Data(
            'n_trials',
            n_trials
        )
        times_data = pm.Data(
            'times',
            times
        )

        # this assumes each ind has is exposed to the same times for the same
        # number of trials
        centered_times = pm.Deterministic(
            'centered_times',
            times_data - pm.math.mean(times_data)
        )

        alpha_mu = pm.Normal(
            'alpha_mu',
            **alpha_mu_kwargs
        )
        alpha_sigma = pm.HalfCauchy(
            'alpha_sigma',
            **alpha_sigma_kwargs
        )
        beta_mu = pm.Normal(
            'beta_mu',
            **beta_mu_kwargs
        )
        beta_sigma = pm.HalfCauchy(
            'beta_sigma',
            **beta_sigma_kwargs
        )

        alpha = pm.Normal(
            'alpha',
            mu=alpha_mu,
            sigma=alpha_sigma,
            shape=(len(np.unique(id_ind)))
        )
        beta = pm.Normal(
            'beta',
            mu=beta_mu,
            sigma=beta_sigma,
            shape=(len(np.unique(id_ind)))
        )

        theta = pm.Deterministic(
            'theta',
            pm.logit(
                alpha[id_ind_data] + (beta[id_ind_data] * centered_times)
            )
        )

        observed_responses = pm.Binomial(
            'observed_responses',
            p=theta,
            n=n_trials_data
        )

    return model
