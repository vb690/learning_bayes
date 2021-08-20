import numpy as np

import pymc3 as pm


def models_comparison_bf(models, comparisons):
    """
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
    """
    """
    bfs = {}
    for model in models.items():

        bfs[model] = model['posterior'] / model['prior']

    return bfs
