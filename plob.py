import numpy as np
from collections import deque
from tqdm import tqdm

from gym.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from gen_state import gen_order_book, gen_player_seq, gen_value
from utils import feasible_action, get_state_lob_after_order


class OrderBookEnv(MultiAgentEnv):
    def __init__(
            self, num_player=2, num_informed=1, grid_first=4, grid_second=8, grid=1, max_depth=2,
            theta=None, sigma_V=1, delay=10, maturity=10000, gamma=0.97, penalty=0 #-0.1
    ):

        if theta is None:
            theta = [0.3, 0.3, 0.4, 0.2, 0.2, 0.2]
        self.theta = theta
        assert type(grid) == int
        self.num_player = num_player
        self.num_informed = num_informed
        self.max_depth = max_depth
        self.delay = delay
        self.sigma_V = sigma_V
        self.agents = ["agent_%s" % i for i in range(self.num_player)]
        self.group_informed = self.agents[:num_informed]
        self.group_front = self.agents[num_informed:]
        self.maturity = maturity
        self.gamma = gamma
        self.done_base = {agent: False for agent in self.agents}
        self.done_base["__all__"] = False
        self.penalty = penalty

        # Possible actions -> [:self.grid]: buy orders, self.grid: no order, [self.grid+1:]: sell orders
        if grid == 0:
            self.grid = grid_first
        elif grid == 1:
            self.grid = grid_second
        else:
            if not type(grid) == int:
                raise ValueError('grid should be integer')
            self.grid = (grid + 1) * 4

        self.action_space = {agent: Discrete(2 * self.grid + 1) for agent in self.agents}
        # Construct state space of limit order book
        self.P, self.state_lob_vec_list = gen_order_book(grid_first=grid_first, grid_second=grid_second,
                                                         grid=grid, max_depth=max_depth)
        self.state_lob_idx_max = num_player ** (self.grid * max_depth) - 1

        # Arrays of player orders & values
        self.player_orders = gen_player_seq(theta, maturity=self.maturity, num_player=num_player,
                                            num_informed=num_informed)
        self.arrive_count = [list(self.player_orders).count(i) for i in range(self.num_player)]
        print("arrive count: ", self.arrive_count)
        self.values, self.values_delayed = gen_value(bounds=[0., max(self.P)], sigma=sigma_V, maturity=self.maturity,
                                                     delay=delay)
        assert len(self.values_delayed) == len(self.values) == len(self.player_orders)

        # observation array: [arrived, *state_lob_vec, values or values_delay, state_lob_idx]
        self.low = np.array([0.] + [-self.max_depth] * self.grid + [0.] * 2)
        self.high = np.array(
            [self.num_player] + [self.max_depth] * self.grid + [max(self.P)] + [self.state_lob_idx_max])
        self.observation_space = {
            agent: Box(low=self.low, high=self.high, shape=(self.grid + 3,), dtype=np.float32) for agent in self.agents
        }

        # set terminal time
        self.maturity_initial = maturity
        self.maturity = maturity

        # set initial conditions
        self.rew_delay = {agent: deque([0] * delay) for agent in self.group_front}
        self.arrived = self.player_orders.popleft()
        self.state_lob_idx = 0

        # transaction number
        self.transaction_num = {agent: 0 for agent in self.agents}
        self.zero_order = {agent: 0 for agent in self.agents}

        # set initial state = [*state_lob_vec (=[0,...,0]), values, values_delay]
        self.state = np.zeros(self.grid + 2)
        self.state[-2] = self.values.popleft()
        self.state[-1] = self.values_delayed.popleft()
        while self.arrived == self.num_player:
            self.arrived = self.player_orders.popleft()
            self.state[-2] = self.values.popleft()
            self.state[-1] = self.values_delayed.popleft()
        assert len(self.values_delayed) == len(self.values) == len(self.player_orders)

    def step(self, action: MultiAgentDict):
        # (1-1) save current state
        temp_player = self.arrived
        # print('current player:', temp_player, self.player_orders)
        assert temp_player != self.num_player

        done = {agent: False for agent in self.agents}
        done["__all__"] = False
        temp_state_lob_idx = self.state_lob_idx
        temp_state_lob_vec = self.state[:self.grid]
        temp_value = self.state[-2]
        temp_value_delay = self.state[-1]

        # count zero transaction
        self.transaction_num = {agent: 0 for agent in self.agents}
        self.zero_order = {agent: 0 for agent in self.agents}

        # set base reward
        if self.penalty != 0:
            reward = {
                agent: 0 if agent == self.agents[temp_player]
                else 0 if action[agent] == self.grid else self.penalty
                for agent in self.agents
            }
        else:
            reward = {agent: 0 for agent in self.agents}

        # set action of the arrived player
        action_arrived = action[self.agents[temp_player]]

        # (1-2) check action feasibility
        if not feasible_action(state_lob_vec=temp_state_lob_vec, order_idx=action_arrived, max_depth=self.max_depth):
            if self.penalty != 0:
                reward[self.agents[temp_player]] = self.penalty
                print('error 2: penalize')
                return self._obs(), reward, self.done_base, self._market_quality_check()
            else:
                action_arrived = self.grid

        # update transaction number
        self.transaction_num[self.agents[temp_player]] += 1
        if action_arrived == self.grid:
            self.zero_order[self.agents[temp_player]] += 1

        # (2-1) state_lob_idx & state_lob_vec after order
        temp_state_lob_idx, temp_state_lob_vec, matched = get_state_lob_after_order(
            player_i=temp_player, num_player=self.num_player, state_lob_vec=temp_state_lob_vec,
            state_lob_idx=temp_state_lob_idx, order_idx=action_arrived, market=False
        )

        # (2-2) Update reward after order
        reward = self._rew(reward, temp_value, action_arrived, matched)

        # (2-3) Update state
        temp_player = self.player_orders.popleft()
        temp_value = self.values.popleft()
        temp_value_delay = self.values_delayed.popleft()
        self.maturity -= 1
        done = self._check_done(done)

        # (3) check next order -> if market order, execute that
        while temp_player == self.num_player and not done["__all__"]:
            # (3-1) if state_lob_vec is zero, pass the market order
            if np.sum(np.abs(temp_state_lob_vec)) == 0:
                temp_player, temp_value, temp_value_delay, reward, done = self._update_state_and_rew(reward, done)
                if done["__all__"]:
                    break
                continue

            # (3-2) Generate market order randomly
            market_order = np.random.randint(2, size=1)[0]
            # Buy market order
            if market_order == 0:
                try:
                    position_market = np.min(np.where(temp_state_lob_vec < 0))
                except ValueError:
                    temp_player, temp_value, temp_value_delay, reward, done = self._update_state_and_rew(reward, done)
                    if done["__all__"]:
                        break
                    continue
                market_order = position_market

            # Sell market order
            else:
                try:
                    position_market = np.max(np.where(temp_state_lob_vec > 0))
                except ValueError:
                    temp_player, temp_value, temp_value_delay, reward, done = self._update_state_and_rew(reward, done)
                    if done["__all__"]:
                        break
                    continue
                market_order = self.grid + position_market + 1

            # (3-3) state_lob_idx, state_lob_vec after market order
            temp_state_lob_idx, temp_state_lob_vec, matched = get_state_lob_after_order(
                player_i=temp_player, num_player=self.num_player, state_lob_vec=temp_state_lob_vec,
                state_lob_idx=temp_state_lob_idx, order_idx=market_order, market=True
            )

            # (3-4) Update reward
            reward = self._rew(reward, temp_value, market_order, matched)

            # next player & value
            done = self._check_done(done)
            if done["__all__"]:
                break
            temp_player = self.player_orders.popleft()
            temp_value = self.values.popleft()
            temp_value_delay = self.values_delayed.popleft()
            self.maturity -= 1

        # Update state
        self.arrived = temp_player
        if temp_player == self.num_player:
            assert done["__all__"] == True
        self.state[-2] = temp_value
        self.state[-1] = temp_value_delay
        self.state[:self.grid] = temp_state_lob_vec
        self.state_lob_idx = temp_state_lob_idx

        # check maturity
        if self.maturity <= 0:
            done["__all__"] = True

        if np.isnan(self.state).any():
            print('nan state!! ', self.state)

        assert not np.isnan(self.state).any()
        done = self._check_done(done)
        # Set info
        info = self._market_quality_check()

        return self._obs(), reward, done, info

    def reset(self):
        # set terminal time
        self.maturity = self.maturity_initial

        # Arrays of player orders & values
        self.player_orders = gen_player_seq(self.theta, maturity=self.maturity,
                                            num_player=self.num_player, num_informed=self.num_informed)
        self.values, self.values_delayed = gen_value(bounds=[0, max(self.P)], sigma=self.sigma_V,
                                                     maturity=self.maturity,
                                                     delay=self.delay)
        # set initial delayed reward
        self.rew_delay = {agent: deque([0] * self.delay) for agent in self.group_front}

        # set transaction number zero
        self.transaction_num = {agent: 0 for agent in self.agents}
        self.zero_order = {agent: 0 for agent in self.agents}

        # set initial state = [*state_lob_vec (=[0,...,0]), state_lob_idx=0, values, values_delay]
        self.arrived = self.player_orders.popleft()
        self.state_lob_idx = 0

        # set initial state = [*state_lob_vec (=[0,...,0]), values, values_delay]
        self.state = np.zeros(self.grid + 2)
        self.state[-2] = self.values.popleft()
        self.state[-1] = self.values_delayed.popleft()
        while self.arrived == self.num_player:
            self.arrived = self.player_orders.popleft()
            self.state[-2] = self.values.popleft()
            self.state[-1] = self.values_delayed.popleft()

        return self._obs()

    def render(self):
        pass

    def _check_done(self, done):
        if done["__all__"]:
           return self._change_done_all(done)
        elif len(self.player_orders) <= 1 or len(self.values) <= 1 or len(self.values_delayed) <= 1:
            done = self._change_done_all(done)
        return done

    def _change_done_all(self, done):
        done = {k: True for k in done}
        done["__all__"] = True
        return done

    def _update_state_and_rew(self, reward, done):
        try:
            temp_player = self.player_orders.popleft()
            temp_value = self.values.popleft()
            temp_value_delay = self.values_delayed.popleft()
            reward = self._rew(reward, temp_value, action=self.grid, matched=None)
            self.maturity -= 1
        except IndexError:
            done["__all__"] = True
            temp_player, temp_value, temp_value_delay = None, None, None
        done = self._check_done(done)
        return temp_player, temp_value, temp_value_delay, reward, done

    def _obs(self):
        # observation array: [*state_lob_vec, values or values_delay, state_lob_idx]
        obs_informed = np.append(np.array([self.arrived]), self.state[:-1])
        obs_front = np.append(np.array([self.arrived]), np.append(self.state[:-2], self.state[-1]))
        obs = {
            agent: np.append(obs_informed, self._get_lob_idx_obs(agent)) if agent in self.group_informed
            else np.append(obs_front, self._get_lob_idx_obs(agent))
            for agent in self.agents
        }
        # for v in obs.values():
        #     if not self._contains(v):
        #         self._contains(v)
        return obs

    def _contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return np.all(x >= self.low) and np.all(x <= self.high)

    def _get_lob_idx_obs(self, agent):
        if self.num_player == 2:
            return self.state_lob_idx

        agent = agent[-1]
        if agent == '0':
            _agent = '1'
        else:
            _agent = '0'

        temp = str(self.state_lob_idx)
        for char in set(temp):
            if char != agent:
                temp.replace(char, _agent)
        temp = int(temp)
        return temp

    # Update reward
    def _rew(self, reward, temp_value, action, matched):
        for agent in self.group_front:
            reward[agent] += self.rew_delay[agent].popleft()
            self.rew_delay[agent].append(0)

        if (matched is None) or (action == self.grid):
            return reward

        matched_agent = self.agents[matched]
        # matched_agent is informed
        if matched < len(self.group_informed):
            if action < self.grid:
                reward[matched_agent] += (temp_value - self.P[action]) / self.arrive_count[0]
            elif action > self.grid:
                reward[matched_agent] += (self.P[action - self.grid - 1] - temp_value) / self.arrive_count[0]
            else:
                pass

        # matched_agent is front-runner
        elif matched < self.num_player:
            if action < self.grid:
                self.rew_delay[matched_agent][-1] += (temp_value - self.P[action]) / self.arrive_count[1]
            elif action > self.grid:
                self.rew_delay[matched_agent][-1] += (self.P[action - self.grid - 1] - temp_value) / self.arrive_count[
                    1]
            else:
                pass
        # matched_agent is a noise trader
        else:
            raise ValueError('Invalid order in the order book!')

        return reward

    def _market_quality_check(self):
        state_lob_vec = self.state[:self.grid]
        try:
            position_ask = np.min(np.where(state_lob_vec < 0))
        except ValueError:
            position_ask = self.grid
        try:
            position_bid = np.max(np.where(state_lob_vec > 0))
        except ValueError:
            position_bid = -1

        # Bid-Ask Spread
        spread = position_ask - position_bid

        # Bid/Ask Depth
        bid_depth, ask_depth = 0, 0
        if position_bid != -1:
            bid_depth = state_lob_vec[position_bid]
        if position_ask != self.grid:
            ask_depth = abs(state_lob_vec[position_ask])

        # Cumulative Depth
        cumulative_buy_depth = sum([order for order in state_lob_vec if order > 0])
        cumulative_sell_depth = abs(sum([order for order in state_lob_vec if order < 0]))

        # Number of Large Limit Orders
        large_order = self.max_depth // 2
        large_buy_limit = sum([1 for order in state_lob_vec if order >= large_order])
        large_sell_limit = sum([1 for order in state_lob_vec if order <= -large_order])

        info = {agent: {"transaction_num": self.transaction_num[agent], "zero_order": self.zero_order[agent]}
                for agent in self.agents}
        info[self.agents[0]] = {
            **info[self.agents[0]],
            **{
                "spread": spread,
                "bid_ask_depth": (bid_depth, ask_depth),
                "cumulative_depth": (cumulative_buy_depth, cumulative_sell_depth),
                "num_large_limit_orders": (large_buy_limit, large_sell_limit)
            }}
        return info


def sample_multi_action(action_space_dict):
    return {key: action_space_dict[key].sample() for key in action_space_dict}


if __name__ == "__main__":

    maturity = 1000
    for grid in [1]:
        env = OrderBookEnv(maturity=maturity, grid=grid, num_player=4, num_informed=2)
        print('observation space: ', env.observation_space)
        for i_episode in tqdm(range(100)):
            observation = env.reset()
            print('players: ', env.player_orders)
            print('values: ', env.values)
            total = {agent: 0 for agent in env.agents}
            done = {'__all__': False}
            while not done['__all__']:
                # print('t: ', t)
                env.render()
                # print('observation_before: ', observation)
                action = sample_multi_action(env.action_space)
                # print('action: ', action)
                observation, reward, done, info = env.step(action)
                # print('observation_after: ', observation)
                # print('reward:', reward)
                total = {agent: total[agent] + reward[agent] for agent in env.agents}
            print('total: ', total)
