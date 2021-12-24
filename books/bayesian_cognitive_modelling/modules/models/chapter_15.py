import pymc3 as pm

import numpy as np


def SIMPLE_model(conditions_id, items_id, obs_responses):
    """
    """
    with pm.model() as model:

        conditions_id_data = pm.Data(
            'conditions_id_data',
            conditions_id
        )
        items_id_data = pm.Data(
            'items_id_data',
            items_id
        )

    return model
