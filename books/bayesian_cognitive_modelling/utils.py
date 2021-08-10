import pymc3 as pm
import arviz as az

import matplotlib.pyplot as plt


def validate_model(model, prpc_kwargs, sampling_kwargs, popc_kwargs):
    """Utility function for visualizing model plate, sampling from the model
    and getting prior and posterior predictive check.

    Args:
        - model: a pymc3 model, model that needs to be validated.
        - prpc_kwargs: a dict, keyword argument for sampling prior predictive.
        - sampling_kwargs: a dict, keyword argument for sampling.
        - popc_kwargs: a dict, keyword argument for sampling posterior
            predictive.

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

    plate = pm.model_to_graphviz(model)

    return plate, prpc, trace, popc


def visualize_samples(observed, prpc, popc):
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
        s=80,
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
        s=80,
        edgecolors='orange'
    )
    axs[1].set_title('Observed Data')

    axs[2].scatter(
        [pos for pos in range(len(observed))],
        popc,
        facecolors='none',
        s=80,
        edgecolors='g',
        linestyle='--',
        vmin=observed.min(),
        vmax=observed.max()
    )
    axs[2].set_title('Posterior Predictive Check')

    return None
