import numpy as np

import pymc as pm


def rate_estimation(obs_freq, n, **beta_kwargs):
    """PyMC3 implementation of rate estimation.

    Args:
        - obs_freq: array of int, observed frequencies.
        - n: int, number of trials that generated obs_freq.
        - **beta_kwargs: keyword argument, parameters of the prior beta
            distribution.

    Returns:
        - model: PyMC3 model for estimating rate p generating obs_freq
    """
    with pm.Model() as model:

        p = pm.Beta(
            'p',
            **beta_kwargs
        )
        freq = pm.Binomial(
            'frequency',
            n=n,
            p=p,
            observed=obs_freq

        )

    return model


def rate_two_groups(obs_freq_1, obs_freq_2, n_1, n_2, shared=False,
                    **beta_kwargs):
    """PyMC3 implementation of 2 groups rate estimation.

    Args:
        - obs_freq_1: array of int, observed frequencies for the first group.
        - obs_freq_2: array of int, observed frequencies for the second group.
        - n_1: int, number of trials that generated obs_freq_1.
        - n_2: int, number of trials that generated obs_freq_2.
        - shared: bool, controls if the estimated rate is shared or not.
        - **beta_kwargs: keyword argument, parameters of the prior beta
            distribution.

    Returns:
        - model: PyMC3 model for estimating rate p generating obs_freq_1 and
            obs_freq_2.
    """
    with pm.Model() as model:

        if shared:
            p = pm.Beta(
                'p',
                **beta_kwargs
            )
            p_1 = p
            p_2 = p
        else:
            p_1 = pm.Beta(
                'p_1',
                **beta_kwargs
            )
            p_2 = pm.Beta(
                'p_2',
                **beta_kwargs
            )
            delta_p = pm.Deterministic(
                'delta_frequencies',
                p_1 - p_2
            )

        freq_1 = pm.Binomial(
            'frequency_1',
            n=n_1,
            p=p_1,
            observed=obs_freq_1

        )

        freq_2 = pm.Binomial(
            'frequency_2',
            n=n_2,
            p=p_2,
            observed=obs_freq_2

        )

    return model


def joint_rate_trials(obs_freq, max_trials, **beta_kwargs):
    """PyMC3 implementation of joint estimation of rate and trials.

    Args:
        - obs_freq: array of int, observed frequencies.
        - max_trials: int, maximum number of trials we want to explore. The
            prior for max_trials will be un-informantive, for each possible
            trial value its probability will be computed as 1/max_trials.
        - **beta_kwargs: keyword argument, parameters of the prior beta
            distribution.

    Returns:
        - model: PyMC3 model for jointly estimating p and trials
            generating obs_freq.
    """
    p_cat = np.array(
        [i/max_trials for i in range(obs_freq.min(), max_trials+1)]
    )
    with pm.Model() as model:

        p = pm.Beta(
            'p',
            **beta_kwargs
        )
        n = pm.Categorical(
            'n',
            p=p_cat
        )
        freq = pm.Binomial(
            'frequency',
            n=n,
            p=p,
            observed=obs_freq

        )

    return model
