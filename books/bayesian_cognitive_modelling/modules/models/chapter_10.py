import pymc3 as pm

import theano.tensor as tt


def estimate_retention_probability(observed_retained, n_items, t_steps,
                                   beta_remembering_kwargs, beta_decay_kwargs,
                                   id_ind=None):
    """
    """
    with pm.Model() as model:

        t_steps_data = pm.Data(
            't_steps',
            t_steps
        )
        if id_ind is not None:
            id_ind_data = pm.Data(
                'id_ind',
                id_ind
            )
            beta = pm.Beta(
                'alpha',
                shape=(len(id_ind), ),
                **beta_decay_kwargs
            )
            alpha = pm.Beta(
                'alpha',
                shape=(len(id_ind), ),
                **beta_decay_kwargs
            )
            decay = pm.Deterministic(
                'exponential_decay',
                pm.math.exp(
                    -alpha[id_ind_data] * t_steps_data
                )
            )
            theta = pm.Deterministic(
                'theta',
                tt.minimum(1, decay + beta[id_ind_data])
            )
        else:
            beta = pm.Beta(
                'alpha',
                **beta_decay_kwargs
            )
            alpha = pm.Beta(
                'alpha',
                **beta_decay_kwargs
            )
            decay = pm.Deterministic(
                'exponential_decay',
                pm.math.exp(
                    -alpha * t_steps_data
                )
            )
            theta = pm.Deterministic(
                'theta',
                tt.minimum(1, decay + beta)
            )

        observed = pm.Binomial(
            'observed_retained',
            n=n_items,
            p=theta,
            observed=observed_retained
        )

    return model


def estimate_retention_probability_hier(observed_retained, n_items, t_steps,
                                        id_ind,
                                        uniform_alpha_remembering_kwargs,
                                        uniform_beta_remembering_kwargs,
                                        uniform_alpha_decay_kwargs,
                                        uniform_beta_decay_kwargs):
    """
    """
    with pm.Model() as model:

        t_steps_data = pm.Data(
            't_steps',
            t_steps
        )
        id_ind_data = pm.Data(
            'id_ind',
            id_ind
        )

    # ######################### HP #######################################

        beta_alpha_hp = pm.Uniform(
            'alpha_remembering',
            **uniform_alpha_remembering_kwargs
        )
        beta_beta_hp = pm.Uniform(
            'beta_remembering',
            **uniform_beta_remembering_kwargs
        )
        alpha_alpha_hp = pm.Uniform(
            'alpha_decay',
            **uniform_alpha_decay_kwargs
        )
        alpha_beta_hp = pm.Uniform(
            'beta_decay',
            **uniform_beta_decay_kwargs
        )

    # ####################################################################

        beta = pm.Beta(
            'beta',
            shape=(len(id_ind), ),
            alpha=beta_alpha_hp,
            beta=beta_beta_hp
        )
        alpha = pm.Beta(
            'alpha',
            shape=(len(id_ind), ),
            alpha=alpha_alpha_hp,
            beta=alpha_beta_hp
        )
        decay = pm.Deterministic(
            'exponential_decay',
            pm.math.exp(
                -alpha[id_ind_data] * t_steps_data
            )
        )
        theta = pm.Deterministic(
            'theta',
            tt.minimum(1, decay + beta[id_ind_data])
        )

        observed = pm.Binomial(
            'observed_retained',
            n=n_items,
            p=theta,
            observed=observed_retained
        )

    return model
