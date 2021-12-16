from scipy.stats import norm

import pymc3 as pm


def signal_detection(obs_hit, obs_false_allarm, noise_trials,
                     signal_trials, signal_gaussian_kwargs,
                     criterion_gaussian_kwargs):
    """
    """
    with pm.Model() as model:

        noise_trials_data = pm.Data(
            'noise_trials',
            noise_trials
        )
        signal_trials_data = pm.Data(
            'signal_trials',
            signal_trials
        )

        signal = pm.Normal(
            'sigmal',
            **signal_gaussian_kwargs
        )
        criterion = pm.Normal(
            'sigmal',
            **signal_gaussian_kwargs
        )

        theta_hit = pm.Deterministic(
            'theta_hit',
            norm.cdf(0.5 * (signal - criterion))
        )
        theta_false_allarm = pm.Deterministic(
            'theta_false_allarm',
            norm.cdf(-0.5 * (signal - criterion))
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
