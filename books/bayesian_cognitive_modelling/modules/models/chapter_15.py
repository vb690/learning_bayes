import numpy as np

import pymc as pm


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
