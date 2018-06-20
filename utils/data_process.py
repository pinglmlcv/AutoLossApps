# __Author__ == "Haowen Xu"
# __Date__ == "06-18-2018"

import numpy as np

def preprocess(ob):
    state = []
    state.append(np.array(ob.layers['#'], dtype=float))
    state.append(np.array(ob.layers['P'], dtype=float))
    state.append(np.array(ob.layers['$'], dtype=float))
    state = np.array(state)
    state = np.transpose(state, (1, 2, 0))
    state = state - 0.5
    return state

