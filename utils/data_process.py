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
    state = state - 0.1
    return state

def vector_preprocess(ob):
    state = []
    player_coords = get_coords(ob.layers['P'])
    reward_coords = get_coords(ob.layers['$'])
    state.append(player_coords[0])
    state.append(player_coords[1])
    state.append(reward_coords[0])
    state.append(reward_coords[1])
    return state

