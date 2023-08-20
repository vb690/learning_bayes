import numpy as np

import pymc as pm


def estimate_pearson_correlation(obs_measurements, lkj_chol_kwargs, mu_kwargs,
                                 sigma_kwargs=None):
    """PyMC3 implementation of pearson correlation estimation.

    Args:
        - obs_measurements: nD array, n observed measurements.
        - lkj_chol_kwargs: dict, keyword arguments for the Cholesky
            decomposition.
        - mu_kwargs: dict, keyword arguments for a normal distrbution.
        - sigma_kwargs: None or dict,  keyword arguments for a HalfCauchy
            distrbution. If specified allow to model measurement error using
            the value of the multivariate normal as input to as many normal
            distrbution as there are measurements.

    Returns
        - model: PyMC3 model, computing the parameters of a multivariate
            normal distribution from which obs_measurement_1 and
            obs_measurement_2 are drawn. Cholesky decomposition
            provide the correlation matrix. If sigmas are provided, measurement
            error can be estimated.
    """
    with pm.Model() as model:

        n = obs_measurements.shape[1]

        chol, corr, stds = pm.LKJCholeskyCov(
            'Cholesky decomposition',
            n=n,
            compute_corr=True,
            **lkj_chol_kwargs
        )
        cov = pm.Deterministic(
            'covariance',
            chol.dot(chol.T)
        )

        mu = pm.Normal(
            'mu',
            shape=n,
            **mu_kwargs
        )

        if sigma_kwargs is None:
            measurements = pm.MvNormal(
                'measurements',
                mu=mu,
                chol=chol,
                observed=obs_measurements
            )
        else:
            latent_measurements = pm.MvNormal(
                'latent_measurements',
                mu=mu,
                chol=chol,
                shape=(n,)
            )
            sigma = pm.HalfCauchy(
                'measurements_error',
                shape=(n,),
                **sigma_kwargs
            )
            measurements = pm.Normal(
                'measurements',
                mu=latent_measurements,
                sigma=sigma,
                observed=obs_measurements
            )

    return model


def estimate_k_coef_agreement(obs_frequencies, **beta_kwargs):
    """PyMC3 implementation of the kappa coefficient of agreement for a binary
    decision process in which a gold standard is compared to a surrogate method
    by means of n comparisons. The k coefficient is the ratio between the
    observed agreement and the agreement it would have occoured by chance.
    This could be paramtrized with a dirichlet prior
    but we will use separate betas for clarity.

    Args:
        - obs_frequencies: 1D array, frequencies of possible agreement
            outcomes: 00, 01, 11, 10.
        - **beta_kwargs: keyword argument, parameters of the prior beta
            distribution.

    Returns:
        - model: a PyMC3 model estimating  the underlying rate of each
            agreement outcome as well as the derived kappa coefficient.
    """
    with pm.Model() as model:

        n = np.sum(obs_frequencies)

        # rate gold standard decides 1
        alpha = pm.Beta(
            'alpha',
            **beta_kwargs
        )
        # rate gold standard decides 0
        alpha_prime = pm.Deterministic(
            '1-alpha',
            1 - alpha
        )
        # rate when surrogate decides 1 and gold standard decides 1
        beta = pm.Beta(
            'beta',
            **beta_kwargs
        )
        # rate when surrogate decides 0 and the gold standard decides 1
        beta_prime = pm.Deterministic(
            '1-beta',
            1 - beta
        )
        # rate when surrogate decides 0 when the gold standard decides 0
        gamma = pm.Beta(
            'gamma',
            **beta_kwargs
        )
        # rate when surrogate decides 1 and the gold standard decides 0
        gamma_prime = pm.Deterministic(
            '1-gamma',
            1 - gamma
        )

        # agreement 11
        p_11 = pm.Deterministic(
            '11',
            alpha*beta
        )
        # agreement 00
        p_00 = pm.Deterministic(
            '00',
            alpha_prime*gamma
        )
        # disagreement 01
        p_10 = pm.Deterministic(
            '10',
            alpha*beta_prime
        )
        # disagreement 10
        p_01 = pm.Deterministic(
            '01',
            alpha_prime*gamma_prime
        )

        # computing k
        total_agreement = pm.Deterministic(
            'total_agreement',
            (alpha*beta) + (alpha_prime*gamma)
        )
        chance_agreement = pm.Deterministic(
            'chance_agreement',
            ((p_11 + p_10) * (p_11 + p_01)) + ((p_10 + p_00) * (p_01 + p_00))
        )
        k = pm.Deterministic(
            'k',
            (total_agreement - chance_agreement) / (1 - chance_agreement)
        )

        # generating observed frequencies from estimated rates of agreement
        obs_frequencies = pm.Multinomial(
            'obs_frequencies',
            n=n,
            p=[p_11, p_00, p_10, p_01],
            observed=obs_frequencies
        )

    return model


def estimate_change_point(obs_time_series, t_steps, slope_kwargs,
                          intercept_kwargs, sigma_kwargs):
    """PyMC3 implementation of a single change point detection.

    Args:
        - obs_time_series: numpy array, values of the time series.
        - t_steps: numpy array, time indices associated with the time
            series.
        - mu_kwargs: dict, keyword argument for a Normal distrbution.
        - sigma_kwargs: dict, keyword arguments for an HalfCauchy distrbution.

    Returns:
        - model: a PyMC3 model, given a stationary time series the model
            detects at which point in time there is a shift in the mu
            parameter of the normal distribution generating the time series.
    """
    with pm.Model() as model:

        lag_1_time_series = pm.Data(
            'lag_1_time_series',
            obs_time_series[:-1]
        )
        time_steps = pm.Data(
            'time_steps',
            t_steps
        )

        slope = pm.Normal(
            'slope',
            **slope_kwargs
        )
        intercepts = pm.Normal(
            'intercepts',
            shape=(2,),
            **intercept_kwargs
        )
        sigma = pm.HalfCauchy(
            'sigma',
            **sigma_kwargs
        )

        change_point = pm.DiscreteUniform(
            'change_point',
            time_steps.min(),
            time_steps.max()
        )
        intercept = pm.math.switch(
            t_steps >= change_point, intercepts[1], intercepts[0]
        )

        mu = pm.Deterministic(
            'mu',
            intercept + slope * lag_1_time_series
        )

        time_series = pm.Normal(
            'observed_time_series',
            mu=mu,
            sigma=sigma,
            observed=obs_time_series[1:]
        )

    return model


def estimate_censored_data(n_censored_attempts, observed_attempt, n,
                           beta_kwargs, lower_bound, upper_bound):
    """PyMC3 implementation of censored data generative model.

    Args:
        - observed_attempt: int, number of correct answers for the observed
            attempt.
        - n: int, number of possible correct answers.
        - beta_kwargs: dict, keyword arguments for a beta distribution.
        - lower_bound: int, lower bound for the unobserved data.
        - upper_bound: int, upper_bound for the unobserved data.

    Returns:
        - model: a PyMC3 model, estimating the underlying probability of giving
            a correct answer with the constrains of observing one succesfull
            attempt over n_censored_attempts + 1.
    """
    with pm.Model() as model:

        def censored_ll(n, n_censored_attempts, p, lower_bound, upper_bound):
            """Log likelyhood for bounded binomial distribution.
            """
            binomial = pm.Binomial.dist(n=n, p=p)
            bounds = np.arange(lower_bound, upper_bound + 1)
            return binomial.logp(bounds).sum() * n_censored_attempts

        p = pm.Beta(
          'p',
          **beta_kwargs
        )

        unobserved = pm.Potential(
          'unobservd',
          censored_ll(
              n=n,
              n_censored_attempts=n_censored_attempts,
              p=p,
              lower_bound=lower_bound,
              upper_bound=upper_bound
          )
        )

        observed_attempt = pm.Binomial(
          'observed',
          n=n,
          p=p,
          observed=observed_attempt
        )

    return model


def estimate_recapture(observed_recaptured, first_sample, second_sample,
                       discrete_uniform_kwargs):
    """PyMC3 implementation of a capture-recapture model.

    Args:
        - observed_recaptured: int, number of recaptured elements from
            first_sample.
        - first_sample: int, number of sampled elements for the first time.
        - second_sample: int, number of sampled elements for the second time.
        - discrete_uniform_kwargs: dict, keyword arguments for a discrete
            uniform distribution.

    Returns:
        - model: a PyMC3 model, it estimates the size of a population from
            which first_sample was drawn and subsequently second_sample was
            drawn observing observed_recaptured elements.
    """
    with pm.Model() as model:

        population_size = pm.DiscreteUniform(
            'population_size',
            **discrete_uniform_kwargs
        )

        recaptured = pm.HyperGeometric(
            'recaptured',
            n=second_sample,
            k=first_sample,
            N=population_size,
            observed=observed_recaptured
        )

    return model
