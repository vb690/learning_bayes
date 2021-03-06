from scipy.stats import sem

import pymc3 as pm
import arviz as az

import matplotlib.pyplot as plt


def validate_model(model, prpc_kwargs, sampling_kwargs, popc_kwargs,
                   show_plate=True):
    """Utility function for visualizing model plate, sampling from the model
    and getting prior and posterior predictive check.

    Args:
        - model: a pymc3 model, model that needs to be validated.
        - prpc_kwargs: a dict, keyword argument for sampling prior predictive.
        - sampling_kwargs: a dict, keyword argument for sampling.
        - popc_kwargs: a dict, keyword argument for sampling posterior
            predictive.
        - show_plate: a bool determine if a plate is returned, patch bug fix
            untill a solution for pymc3 impossibility to graph HyperGeometric
            distr is found.

    Returns:
        - plate:  graphviz Digraph, graphical representation of model.
        - prpc: a dict, prior samples.
        - trace: multitrace object, traces obtained after sampling.
        - popc: a dict, posterior samples.
    """
    with model:

        prpc = pm.sample_prior_predictive(**prpc_kwargs)
        trace = pm.sample(**sampling_kwargs)
        popc = pm.sample_posterior_predictive(
            trace,
            **popc_kwargs
        )

        az.plot_trace(trace)

    if show_plate:
        plate = pm.model_to_graphviz(model)
        return plate, prpc, trace, popc
    else:
        return prpc, trace, popc


def visualize_samples(observed, prpc, popc, s=80):
    """Utility function for visualizing predictions from the prior, posterior
    and actual data.

    Args:
        - observed: array, observed data.
        - prpc: array, sample from the prior predictive.
        - popc: array, sample from the posterior predictive.

    Returns:
        - None
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].scatter(
        [pos for pos in range(len(observed))],
        prpc,
        facecolors='none',
        s=s,
        edgecolors='b',
        linestyle='--',
        vmin=observed.min(),
        vmax=observed.max()
    )
    axs[0].set_title('Prior Predictive Check')

    axs[1].scatter(
        [pos for pos in range(len(observed))],
        observed,
        facecolors='none',
        s=s,
        edgecolors='orange'
    )
    axs[1].set_title('Observed Data')

    axs[2].scatter(
        [pos for pos in range(len(observed))],
        popc,
        facecolors='none',
        s=s,
        edgecolors='g',
        linestyle='--',
        vmin=observed.min(),
        vmax=observed.max()
    )
    axs[2].set_title('Posterior Predictive Check')

    return None


def visualize_biv_samples(observed, prpc, popc, s=80):
    """Utility function for visualizing bivariate,
    predictions from the prior, posterior and actual data as bivariate

    Args:
        - observed: array, observed data.
        - prpc: array, sample from the prior predictive.
        - popc: array, sample from the posterior predictive.

    Returns:
        - None
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].scatter(
        prpc[:, 0],
        prpc[:, 1],
        facecolors='none',
        s=s,
        edgecolors='b',
        linestyle='--',
        vmin=observed.min(),
        vmax=observed.max()
    )
    axs[0].set_title('Prior Predictive Check')

    axs[1].scatter(
        observed[:, 0],
        observed[:, 1],
        facecolors='none',
        s=s,
        edgecolors='orange'
    )
    axs[1].set_title('Observed Data')

    axs[2].scatter(
        popc[:, 0],
        popc[:, 1],
        facecolors='none',
        s=s,
        edgecolors='g',
        linestyle='--',
        vmin=observed.min(),
        vmax=observed.max()
    )
    axs[2].set_title('Posterior Predictive Check')

    return None


def visualize_time_series(observed, prpc, popc, change_points=None,
                          figsize=(15, 5)):
    """Utility function for visualizing time series with or without change
    point.

    Args:
        - observed: array, observed data.
        - prpc: array, samples from the prior predictive.
        - popc: array, samples from the posterior predictive.
        - change_point: if not None, list of change_points. This indicate the
            observed and estimated change point in the time series.
        - figsize: tuple, size of the figure canvas.

    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # prior
    axs[0].plot(
        prpc.mean(axis=0),
        linestyle='--',
        c='b'
    )
    axs[0].fill_between(
        [t for t in range(len(observed))],
        prpc.mean(axis=0) + sem(prpc, axis=0),
        prpc.mean(axis=0) - sem(prpc, axis=0),
        color='b',
        alpha=0.25
    )
    axs[0].set_title('Prior Predictive Check')
    if change_points is not None:
        for sample in range(change_points[0].shape[0]):

            axs[0].axvline(
                change_points[0][sample],
                c='r',
                alpha=0.01
            )

    # observed data
    axs[1].plot(
        observed,
        c='orange'
    )
    if change_points is not None:
        axs[1].axvline(
            change_points[1],
            c='r',
        )
    axs[1].set_title('Observed Data')

    # posterior
    axs[2].plot(
        popc.mean(axis=0),
        linestyle='--',
        c='g'
    )
    axs[2].fill_between(
        [t for t in range(len(observed))],
        popc.mean(axis=0) + sem(popc, axis=0),
        popc.mean(axis=0) - sem(popc, axis=0),
        color='g',
        alpha=0.25
    )
    if change_points is not None:
        for sample in range(change_points[2].shape[0]):

            axs[2].axvline(
                change_points[2][sample],
                c='r',
                alpha=0.01
            )
    axs[2].set_title('Posterior Predictive Check')

    return None


def visualize_matrices(observed, prpc, popc, figsize=(15, 5)):
    """Utility function for visualizing matrices
    Args:
        - observed: array, observed data.
        - prpc: array, sample from the prior predictive.
        - popc: array, sample from the posterior predictive.

    Returns:
        - None
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    axs[0].matshow(
        prpc,
        cmap='viridis',
    )
    axs[0].set_title('Prior Predictive Check')

    axs[1].matshow(
        observed,
        cmap='binary',
    )
    axs[1].set_title('Observed Data')

    axs[2].matshow(
        popc,
        cmap='magma',
    )
    axs[2].set_title('Posterior Predictive Check')

    return None
