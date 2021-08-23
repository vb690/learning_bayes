import numpy as np


def models_comparison_bf(models, comparisons):
    """Computing the BF given a set of models.

    Args:
        - models: a dict, key are models name values are pymc3 models after
            sampling.
        - comparisons: a list of tuples, each tuple contain the name of the
            models to be compared.

    Returns:
        - bfs: a dictionary, bayes factor associated to each comparison
    """
    bfs = {}
    for (m_1, m_0) in comparisons:

        m_1_mll = models[m_1].report.log_marginal_likelihood
        m_0_mll = models[m_0].report.log_marginal_likelihood

        bf10 = np.exp(
            m_1_mll - m_0_mll
        )

        bfs[f'{m_1}_{m_0}'] = bf10

    return bfs


def hp_testing_savage_dickey_ratio(models):
    """Computing the BF of a nested model, meaning computing the bf for the
    "null" hypothesis of a given model.

    Args:
        - models: a dict, keys are models names while values are dictionaries
            reporting prior and posterior probabilities for the "null".

    Returns:
        - bfs: a dictionary, bayes factor of the null for each model.
    """
    bfs = {}
    for model_name, model_dict in models.items():

        bfs[model_name] = model_dict['posterior'] / model_dict['prior']

    return bfs
