import numpy as np

from scipy.stats import norm

import pymc3 as pm


def signal_detection(id_ind, obs_hit, obs_false_allarm, noise_trials,
                     signal_trials, signal_gaussian_kwargs,
                     criterion_gaussian_kwargs):
    """
    """
    with pm.Model() as model:

        id_ind_data = pm.Data(
            'id_ind',
            id_ind
        )
        noise_trials_data = pm.Data(
            'noise_trials',
            noise_trials
        )
        signal_trials_data = pm.Data(
            'signal_trials',
            signal_trials
        )

        signal_strength = pm.Normal(
            'signal_strength',
            shape=(len(id_ind),),
            **signal_gaussian_kwargs
        )
        criterion = pm.Normal(
            'criterion',
            shape=(len(id_ind),),
            **signal_gaussian_kwargs
        )

        theta_hit = pm.Deterministic(
            'theta_hit',
            norm.cdf(
                (signal_strength[id_ind_data] * 0.5) - criterion[id_ind_data]
            )
        )
        theta_false_allarm = pm.Deterministic(
            'theta_false_allarm',
            norm.cdf(
                (signal_strength[id_ind_data] * -0.5) - criterion[id_ind_data]
            )
        )

        hit = pm.Binomial(
            'hit',
            p=theta_hit,
            n=signal_trials_data,
            observed=obs_hit
        )
        false_allarm = pm.Binomial(
            'false_allarm',
            p=theta_false_allarm,
            n=noise_trials_data,
            observed=obs_false_allarm
        )

    return model


def hier_signal_detection(id_ind, obs_hit, obs_false_allarm, noise_trials,
                          signal_trials,
                          signal_mu_kwargs,
                          signal_sigma_kwargs,
                          criterion_mu_kwargs,
                          criterion_sigma_kwargs,
                          ):
    """
    """
    with pm.Model() as model:

        id_ind_data = pm.Data(
            'id_ind',
            id_ind
        )
        noise_trials_data = pm.Data(
            'noise_trials',
            noise_trials
        )
        signal_trials_data = pm.Data(
            'signal_trials',
            signal_trials
        )

        # signal
        signal_strength_mu = pm.Norma(
            'signal_strength_mu',
            **signal_mu_kwargs
        )
        signal_strength_sigma = pm.HalfCauchy(
            'signal_strength_sigma',
            **criterion_sigma_kwargs
        )
        signal_strength = pm.Normal(
            'signal_strength',
            mu=signal_strength_mu,
            sigma=signal_strength_sigma,
            shape=(len(id_ind),),
        )

        # criterion
        criterion_mu = pm.Norma(
            'criterion_mu',
            **criterion_mu_kwargs
        )
        criterion_sigma = pm.HalfCauchy(
            'criterion_sigma',
            **criterion_sigma_kwargs
        )
        criterion = pm.Normal(
            'criterion',
            shape=(len(id_ind),),
            mu=criterion_mu,
            sigma=criterion_sigma,
        )

        theta_hit = pm.Deterministic(
            'theta_hit',
            norm.cdf(
                (signal_strength[id_ind_data] * 0.5) - criterion[id_ind_data]
            )
        )
        theta_false_allarm = pm.Deterministic(
            'theta_false_allarm',
            norm.cdf(
                (signal_strength[id_ind_data] * -0.5) - criterion[id_ind_data]
            )
        )

        hit = pm.Binomial(
            'hit',
            p=theta_hit,
            n=signal_trials_data,
            observed=obs_hit
        )
        false_allarm = pm.Binomial(
            'false_allarm',
            p=theta_false_allarm,
            n=noise_trials_data,
            observed=obs_false_allarm
        )

    return model


def non_centered_signal_detection(id_ind, obs_hit, obs_false_allarm,
                                  noise_trials, signal_trials,
                                  signal_mu_kwargs, criterion_mu_kwargs,
                                  signal_lamb_kwargs, criterion_lamb_kwargs,
                                  signal_eta_kwargs, criterion_eta_kwargs,
                                  ):
    """
    """
    with pm.Model() as model:

        id_ind_data = pm.Data(
            'id_ind',
            id_ind
        )
        noise_trials_data = pm.Data(
            'noise_trials',
            noise_trials
        )
        signal_trials_data = pm.Data(
            'signal_trials',
            signal_trials
        )

        # signal
        signal_strength_mu = pm.Normal(
            'signal_strength_mu',
            **signal_mu_kwargs
        )
        signal_strength_lambda = pm.HalfCauchy(
            'signal_strength_lambda',
            **signal_lamb_kwargs
        )
        signal_strength_eta = pm.Beta(
            'signal_strength_eta',
            **signal_eta_kwargs
        )
        signal_strength_delta = pm.Normal(
            'signal_strength_nomral',
            mu=0,
            sigma=signal_strength_lambda,
            shape=(len(id_ind),),
        )
        signal_strength_sigma = pm.Deterministic(
            'signal_strength_sigma',
            1 / np.sqrt(signal_strength_lambda)
        )
        signal_strength = pm.Deterministic(
            'signal_strength',
            signal_strength_mu[id_ind_data] + (
                signal_strength_eta*signal_strength_delta
            )
        )

        # criterion
        criterion_mu = pm.Normal(
            'criterion_mu',
            **criterion_mu_kwargs
        )
        criterion_lambda = pm.HalfCauchy(
            'criterion_lambda',
            **criterion_lamb_kwargs
        )
        criterion_eta = pm.Beta(
            'criterionh_eta',
            **criterion_eta_kwargs
        )
        criterion_delta = pm.Normal(
            'criterion_nomral',
            mu=0,
            sigma=criterion_lambda,
            shape=(len(id_ind),),
        )
        criterionh_sigma = pm.Deterministic(
            'criterion_sigma',
            1 / np.sqrt(criterion_lambda)
        )
        criterion = pm.Deterministic(
            'criterion',
            criterion_mu[id_ind_data] + (
                criterion_eta*criterion_delta
            )
        )

        theta_hit = pm.Deterministic(
            'theta_hit',
            norm.cdf(
                (signal_strength[id_ind_data] * 0.5) - criterion[id_ind_data]
            )
        )
        theta_false_allarm = pm.Deterministic(
            'theta_false_allarm',
            norm.cdf(
                (signal_strength[id_ind_data] * -0.5) - criterion[id_ind_data]
            )
        )

        hit = pm.Binomial(
            'hit',
            p=theta_hit,
            n=signal_trials_data,
            observed=obs_hit
        )
        false_allarm = pm.Binomial(
            'false_allarm',
            p=theta_false_allarm,
            n=noise_trials_data,
            observed=obs_false_allarm
        )

    return model
