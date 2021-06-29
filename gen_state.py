import numpy as np
import itertools
from tqdm import tqdm
import random
from collections import deque
from symbulate import BoxModel, RV, Normal
import copy
from utils import validity_state_lob


def gen_order_book(grid_first=4, grid_second=8, grid=1, max_depth=2, get_list=False):

    # Price grid
    if grid == 0:
        price_num = grid_first
    elif grid == 1:
        price_num = grid_second
    else:
        if not type(grid) == int:
            raise ValueError('grid should be integer')
        price_num = (grid + 1) * 4

    price_min = 0
    price_max = (grid_first - 1) * (grid_second - 1)
    P = np.linspace(price_min, price_max, price_num)
    print('Prices: ', P)

    if get_list:
        # Generate states
        state_lob_vec_list = range(-max_depth, max_depth + 1)
        state_lob_vec_list = np.array(
            [order for order in tqdm(itertools.product(state_lob_vec_list, repeat=price_num)) if
             validity_state_lob(order)],
        )
        print('State Limit Order Book Vector list: ', state_lob_vec_list)
    else:
        state_lob_vec_list = None

    return P, state_lob_vec_list


def gen_player_seq(theta, maturity, num_player, num_informed):
    assert len(theta) == 6
    results = _gen_player_seq_helper(theta, maturity, num_player, num_informed)
    while len(results) < maturity:
        temp = _gen_player_seq_helper(theta, maturity, num_player, num_informed)
        results.extend(temp)
        results = results[:maturity]
    return deque(results)


def _gen_player_seq_helper(theta, maturity, num_player, num_informed):
    assert len(theta) == 6
    theta_1, theta_2, theta_3, theta_12, theta_13, theta_23 = theta

    p_000 = np.exp(-sum(theta))
    p_100 = theta_1 * p_000
    p_010 = theta_2 * p_000
    p_001 = theta_3 * p_000
    p_110 = theta_1 * p_010 + theta_12 * p_000
    p_101 = theta_1 * p_001 + theta_13 * p_000
    p_011 = theta_2 * p_001 + theta_23 * p_000
    p_111 = theta_1 * p_011 + theta_12 * p_001 + theta_13 * p_010
    all = p_000 + p_100 + p_010 + p_001 + p_110 + p_101 + p_011 + p_111
    probs = [p_000 / all, p_100 / all, p_010 / all, p_001 / all, p_110 / all, p_101 / all, p_011 / all, p_111 / all]

    player = [0, 1, 2, 3, 4, 5, 6, 7]
    sim = BoxModel(player, probs=probs).sim(maturity)
    sim = [sim.get(i) for i in range(maturity)]

    results = []
    for s in sim:
        if s == 0:
            temp = []
        elif (s == 1) or (s == 2) or (s == 3):
            temp = [s - 1]
        elif s == 4:
            temp = [0, 1]
        elif s == 5:
            temp = [0, 2]
        elif s == 6:
            temp = [1, 2]
        else:
            temp = [0, 1, 2]
        random.shuffle(temp)
        results.extend(temp)

    # Todo multiple informed & front
    if num_player > 2:
        num_front = num_player - num_informed
        result2 = copy.deepcopy(results)
        results = []
        informed = [i for i in range(num_informed)]
        front = [i + num_informed for i in range(num_front)]
        for s in result2:
            if s == 0:
                temp = random.choice(informed)
            elif s == 1:
                temp = random.choice(front)
            else:
                temp = num_player
            results.append(temp)
            if len(results) > maturity:
                break
    return results[:maturity]


def gen_value(bounds, sigma, maturity, delay):
    p_low, p_high = bounds
    v_0 = (p_low + p_high)/2
    dv = RV(Normal(mean=0, sd=sigma)).sim(maturity)
    dv = [dv.get(i) for i in range(maturity)]

    values = [v_0]
    for i in range(maturity-1):
        value = max(min(values[i] + dv[i], p_high), p_low)
        values.append(value)

    values_delay = [v_0]*delay
    values_delay.extend(values[:-delay])
    assert len(values) == len(values_delay) == maturity
    assert np.all(np.array(values) <= p_high) and np.all(np.array(values) >= p_low)
    return deque(values), deque(values_delay)


if __name__ == "__main__":
    input = gen_order_book()

