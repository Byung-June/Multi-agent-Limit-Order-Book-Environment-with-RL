import numpy as np
import itertools
import string


def get_order_from_idx(price_grid, order_idx):
    assert 0 <= order_idx <= 2*price_grid
    if order_idx == price_grid:
        return np.zeros(price_grid, dtype=int)
    elif order_idx < price_grid:
        return np.eye(price_grid, dtype=int)[order_idx]
    elif order_idx > price_grid:
        return -np.eye(price_grid, dtype=int)[order_idx-price_grid-1]
    else:
        raise ValueError("Check get_order_from_idx")


def convert(num, base):
    tmp = string.digits + string.ascii_lowercase
    q, r = divmod(num, base)
    if q == 0:
        return tmp[r]
    else:
        return convert(q, base) + tmp[r]


def get_state_lob_after_order(player_i, num_player, state_lob_vec, state_lob_idx, order_idx, market):
    price_grid = len(state_lob_vec)
    matched = None
    if order_idx == price_grid:
        return state_lob_idx, state_lob_vec, None
    # if not num_player ** np.sum(np.abs(state_lob_vec)) > state_lob_idx:
    #     print(num_player ** np.sum(np.abs(state_lob_vec)), state_lob_idx,
    #                          num_player ** np.sum(np.abs(state_lob_vec)) > state_lob_idx)
    #     raise AssertionError(num_player ** np.sum(np.abs(state_lob_vec)), state_lob_idx,
    #                          num_player ** np.sum(np.abs(state_lob_vec)) > state_lob_idx)
    assert not ((player_i == num_player) and not market)

    # # For debugging,
    order = get_order_from_idx(price_grid=len(state_lob_vec), order_idx=order_idx)
    state_lob_vec_after = state_lob_vec + order
    # print('order: ', order_idx, order, market, player_i)
    # print('before action ', state_lob_vec, convert(state_lob_idx, num_player), state_lob_idx)

    try:
        # order is sell order
        if order_idx > price_grid:
            order_idx = order_idx - price_grid - 1
            # pop order
            if np.sign(state_lob_vec[order_idx]) == 1:
                cum = np.abs(state_lob_vec)[::-1].cumsum()[::-1]
                front, back = divmod(state_lob_idx, num_player ** (cum[order_idx] - 1))

                state_lob_idx_after = int((front // num_player) * num_player ** (cum[order_idx] - 1) + back)
                matched = int(front % num_player)

                # if not num_player ** np.sum(np.abs(state_lob_vec_after)) > state_lob_idx_after:
                #     print(state_lob_idx_after)
            # append order
            else:
                cum2 = np.abs((np.append(state_lob_vec, 0))[1:])[::-1].cumsum()[::-1]
                front, back = divmod(state_lob_idx, num_player ** cum2[order_idx])
                state_lob_idx_after = int(front * num_player ** (cum2[order_idx] + 1)
                                    + player_i * num_player ** cum2[order_idx] + back)

                # if not num_player ** np.sum(np.abs(state_lob_vec_after)) > state_lob_idx_after:
                #     print(state_lob_idx_after)
        # order is buy order
        else:
            # pop order
            if np.sign(state_lob_vec[order_idx]) == -1:
                cum = np.abs(state_lob_vec)[::-1].cumsum()[::-1]
                front, back = divmod(state_lob_idx, num_player ** (cum[order_idx] - 1))
                state_lob_idx_after = int((front // num_player) * num_player ** (cum[order_idx] - 1) + back)
                matched = int(front % num_player)

                # if not num_player ** np.sum(np.abs(state_lob_vec_after)) > state_lob_idx_after:
                #     print(state_lob_idx_after)
            # append order
            else:
                cum2 = np.abs((np.append(state_lob_vec, 0))[1:])[::-1].cumsum()[::-1]
                front, back = divmod(state_lob_idx, num_player ** cum2[order_idx])
                state_lob_idx_after = int(front * num_player ** (cum2[order_idx] + 1)
                                    + player_i * num_player ** cum2[order_idx] + back)

                # if not num_player ** np.sum(np.abs(state_lob_vec_after)) > state_lob_idx_after:
                #     print(state_lob_idx_after)
    except TypeError:
        state_lob_idx_after = state_lob_idx

    # if not num_player ** np.sum(np.abs(state_lob_vec_after)) > state_lob_idx_after:
    #     print(state_lob_idx_after)
    # # For debugging,
    # print('after action ', state_lob_vec_after, 'idx: ', convert(state_lob_idx_after, num_player), state_lob_idx_after)

    return state_lob_idx_after, state_lob_vec_after, matched


def feasible_action(state_lob_vec, order_idx, max_depth):
    """
    :param state_lob_vec:
    :param order_idx:
        if order_idx in [0, price_grid): buy orders
        if order_idx == price_grid: no order
        if order_idx in [price_grid+1, 2*price_grid+1]: sell orders
    :return: feasibility (True of False)
    """

    state_lob_after = state_lob_vec + get_order_from_idx(len(state_lob_vec), order_idx)
    if np.max(np.abs(state_lob_after)) > max_depth:
        return False

    # create a sign array of state_lob_vec
    state_lob_vec = np.sign(state_lob_vec)
    price_grid = len(state_lob_vec)

    if order_idx == price_grid:
        return True

    # number of possible buy orders -> position 0 to ask position
    try:
        position_ask = np.min(np.where(state_lob_vec == -1))
    except ValueError:
        position_ask = price_grid

    if position_ask < order_idx < price_grid:
        return False

    # number of possible sell orders -> bid position to last position
    try:
        position_bid = np.max(np.where(state_lob_vec == 1))
    except ValueError:
        position_bid = 0

    if price_grid < order_idx < price_grid + position_bid:
        return False

    return True


def get_val_from_info(information, num_informed):
    dict1 = list(information.values())[0]
    spread = dict1["spread"]
    bid_depth, ask_depth = dict1["bid_ask_depth"]
    cumulative_bid_depth, cumulative_ask_depth = dict1["cumulative_depth"]
    large_buy_limit, large_sell_limit = dict1["num_large_limit_orders"]
    transaction_num1 = sum([i['transaction_num'] for i in list(information.values())[:num_informed]])
    transaction_num2 = sum([i['transaction_num'] for i in list(information.values())[num_informed:]])
    zero_order1 = sum([i['zero_order'] for i in list(information.values())[:num_informed]])
    zero_order2 = sum([i['zero_order'] for i in list(information.values())[num_informed:]])
    return [
        transaction_num1, transaction_num2, zero_order1, zero_order2,
        spread, bid_depth, ask_depth, cumulative_bid_depth, cumulative_ask_depth,
        large_buy_limit, large_sell_limit
    ]

########################################################################
def validity_state_lob(state_lob):
    state_lob = np.sign(state_lob)
    try:
        position_ask = np.min(np.where(state_lob == -1))
        position_bid = np.max(np.where(state_lob == 1))
    except ValueError:
        return True

    if position_bid >= position_ask:
        return False
    else:
        return True


def convert_iter_to_list(_iter_list):
    return [list(_tuple) for _tuple in list(_iter_list)]


def gen_cummulative_orders(_order_vec, num_player):
    players = range(num_player)
    _orders = [convert_iter_to_list(itertools.product(players, repeat=abs(_order))) for _order in _order_vec]
    _orders = convert_iter_to_list(itertools.product(*_orders))
    return _orders


def gen_dict(_array):
    return {i: elt for i, elt in enumerate(_array)}


def get_key_from_dict(dictionary, val):
    for key, value in dictionary.items():
        try:
            if (val == value).all():
                return key
        except ValueError:
            if val == value:
                return key
    return False


def get_idx_from_array(array, val):
    for idx, value in enumerate(array):
        if val == value:
            return idx


def validity_lob_vec_transition(state1, state2):
    # sign check
    if np.min(np.sign(state1) * np.sign(state2)) == -1:
        return False

    # negative change
    temp = [1 for a in np.abs(state1) - np.abs(state2) if not (0 <= a)]
    if temp:
        return False
    else:
        # price priority
        temp = np.sign(state1)
        diff = np.array(state1 - state2)

        try:
            diff_highest_ask = np.max(np.where(np.sign(diff) == -1))
            position_ask = np.min(np.where(temp == -1))
            if (state1[position_ask:diff_highest_ask] != diff[position_ask:diff_highest_ask]).any():
                return False
        except ValueError:
            pass

        try:
            diff_lowest_bid = np.min(np.where(np.sign(diff) == 1))
            position_bid = np.max(np.where(temp == 1))
            if (state1[diff_lowest_bid + 1:position_bid + 1] != diff[diff_lowest_bid + 1:position_bid + 1]).any():
                return False
        except ValueError:
            pass

        return True


def pop_order(state1, diff):
    return [s1[d:] for s1, d in zip(state1, diff)]


########################################################################
if __name__ == "__main__":
    num_player = 2
    state_lob_vec = [3, 2, 0, -2]
    state_lob_idx = 11
    order_idx = 1
    result = get_state_lob_idx_after_order(1, num_player, state_lob_vec, state_lob_idx, order_idx)    # list1 = [1, 2, 3, 1]
    print(result)
    # list2 = [1, 3]
    # print(validity_order_at_price(list1, list2))

    # state1 = np.array([[1, 1, 1],[1, 1, 1], [1, 1, 1], [1, 1, 2]])
    # state2 = np.array([[1, 1, 1],[1, 1, 1], [], []])
    # print(validity_state_transition(state1, state2))
