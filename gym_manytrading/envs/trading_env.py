import gym
from gym import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    Noop = 0
    Liquidate = 1
    Sell = 2
    Buy = 3


class Order:

    def __init__(self, price, action, tick):
        self.price = price
        self.action = action
        self.tick = tick


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, frame_bound, initial_capital=1000):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.frame_bound = frame_bound
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # porfolio
        self.equity = initial_capital
        self.equity_history = (len(self.prices) - 1) * [None]

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # orders
        self._actions = (len(self.prices) - 1) * [None]
        self._last_order = None

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._first_rendering = None
        self.history = None

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def step(self, action):

        if self._done:
            return

        action = Actions(action)

        net_position_profit = None

        self._actions[self._current_tick] = action

        if action == Actions.Buy:
            net_position_profit = self._liquidate()
            self._buy()
        elif action == Actions.Sell:
            net_position_profit = self._liquidate()
            self._sell()
        elif action == Actions.Liquidate:
            net_position_profit = self._liquidate()

        self._calculate_portfolio_equity(net_position_profit)

        observation = self._get_observation()

        info = dict(
            total_reward=0
        )

        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        return observation, net_position_profit, self._done, info

    def _liquidate(self):
        if self._last_order is None\
                or self._last_order.action == Actions.Liquidate\
                or self._last_order.action == Actions.Noop:
            return

        order = Order(self.prices[self._current_tick], Actions.Liquidate, self._current_tick)

        step_reward = self._calculate_position_profit(self._last_order, order)

        self._last_order = order

        return step_reward

    def _sell(self):
        self._last_order = Order(self.prices[self._current_tick], Actions.Sell, self._current_tick)

    def _buy(self):
        self._last_order = Order(self.prices[self._current_tick], Actions.Buy, self._current_tick)

    def _calculate_portfolio_equity(self, net_position_profit):
        if net_position_profit is None and self._last_order is not None:
            net_position_profit = self._calculate_position_profit(self._last_order,
                                                                  Order(self.prices[self._current_tick],
                                                                        Actions.Liquidate,
                                                                        self._current_tick))

        if net_position_profit is None:
            return

        self.equity += net_position_profit
        self.equity_history[self._current_tick] = self.equity

    def _calculate_position_profit(self, open_position, close_position):
        gross_profit = 0

        if open_position.action == Actions.Sell:
            gross_profit = (close_position.price - open_position.price) * self.equity
        elif open_position.action == Actions.Buy:
            gross_profit = (open_position.price - close_position.price) * self.equity

        return gross_profit

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    def render(self, mode="human"):
        pass

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._actions))
        plt.plot(self.prices)

        buy_ticks = []
        sell_ticks = []
        liquidate_ticks = []
        noop_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._actions[i] == Actions.Buy:
                buy_ticks.append(tick)
            elif self._actions[i] == Actions.Sell:
                sell_ticks.append(tick)
            elif self._actions[i] == Actions.Liquidate:
                liquidate_ticks.append(tick)
            elif self._actions[i] == Actions.Noop:
                noop_ticks.append(tick)

        plt.plot(buy_ticks, self.prices[buy_ticks], 'ro')
        plt.plot(sell_ticks, self.prices[sell_ticks], 'go')
        plt.plot(liquidate_ticks, self.prices[liquidate_ticks], 'bo')

        plt.suptitle(
            "Total Reward: %.6f" % self.equity
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

