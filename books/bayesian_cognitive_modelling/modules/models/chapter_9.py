import numpy as np

import pymc3 as pm


def estimate_proportions_with(observed_answers, trials_n,
                              half_cauchy_sigma_rate_kwargs,
                              half_normal_mu_rate_kwargs,
                              half_cauchy_sigma_diff_kwargs,
                              half_normal_mu_size_kwargs):
    """
    """
    grp_1_answers = observed_answers[0]
    grp_2_answers = observed_answers[1]

    grp_1_trials = trials_n[0]
    grp_2_trials = trials_n[1]

    with pm.Model() as model:

        mu_rate = pm.HalfNormal(
            'mu_rate',
            **half_normal_mu_rate_kwargs
        )
        sigma_rate = pm.HalfCauchy(
            'sigma_rate',
            **half_cauchy_sigma_rate_kwargs
        )

        size_diff = pm.HalfNormal(
            'size_diff',
            **half_normal_mu_size_kwargs
        )
        sigma_diff = pm.HalfCauchy(
            'sigma_rate',
            **half_cauchy_sigma_diff_kwargs
        )
        mu_diff = pm.Deterministic(
            'mu_diff',
            size_diff * sigma_diff
        )
        diff = pm.Normal(
            'diff',
            mu=mu_diff,
            sigma=sigma_diff,
            shape=(len(grp_1_answers), )
        )

        phi_grp_1 = pm.Normal(
            'phi_grp_1',
            mu=mu_rate,
            sigma=sigma_rate,
            shape=(len(grp_1_answers), )
        )
        phi_grp_2 = pm.Deterministic(
            'phi_grp_2',
            phi_grp_1 + diff
        )

        theta_grp_1 = pm.Deterministic(
            'theta_grp_1',
            pm.sigmoid(
                phi_grp_1
            )
        )
        theta_grp_2 = pm.Deterministic(
            'theta_grp_2',
            pm.sigmoid(
                phi_grp_2
            )
        )

        observed_grp_1 = pm.Binomial(
            p=theta_grp_1,
            n=grp_1_trials,
            observed=grp_1_answers
        )
        observed_grp_2 = pm.Binomial(
            p=theta_grp_2,
            n=grp_2_trials,
            observed=grp_2_answers
        )

    return model


def estimate_proportions_betw(observed_answers, trials_n,
                              half_cauchy_sigma_rate_kwargs,
                              normal_mu_rate_kwargs,
                              half_normal_mu_size_kwargs):
    """
    """
    grp_1_answers = observed_answers[0]
    grp_2_answers = observed_answers[1]

    grp_1_trials = trials_n[0]
    grp_2_trials = trials_n[1]

    with pm.Model() as model:

        mu_rate = pm.HalfNormal(
            'mu_rate',
            **normal_mu_rate_kwargs
        )
        sigma_rate = pm.HalfCauchy(
            'sigma_rate',
            **half_cauchy_sigma_rate_kwargs
        )

        size_diff = pm.HalfNormal(
            'size_diff',
            **half_normal_mu_size_kwargs
        )
        mu_diff = pm.Deterministic(
            'mu_diff',
            size_diff * sigma_rate
        )

        phi_grp_1 = pm.Normal(
            'phi_grp_1',
            mu=mu_rate + mu_diff,
            sigma=sigma_rate,
            shape=(len(grp_1_answers), )
        )
        phi_grp_2 = pm.Normal(
            'phi_grp_2',
            mu=mu_rate - mu_diff,
            sigma=sigma_rate,
            shape=(len(grp_2_answers), )
        )

        theta_grp_1 = pm.Deterministic(
            'theta_grp_1',
            pm.sigmoid(
                phi_grp_1
            )
        )
        theta_grp_2 = pm.Deterministic(
            'theta_grp_2',
            pm.sigmoid(
                phi_grp_2
            )
        )

        observed_grp_1 = pm.Binomial(
            p=theta_grp_1,
            n=grp_1_trials,
            observed=grp_1_answers
        )
        observed_grp_2 = pm.Binomial(
            p=theta_grp_2,
            n=grp_2_trials,
            observed=grp_2_answers
        )

    return model
