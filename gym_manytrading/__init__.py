from gym.envs.registration import register
from copy import deepcopy

from . import datasets


register(
    id='sample-v0',
    entry_point='gym_manytrading.envs:TradingEnv',
    kwargs={
        'df': deepcopy(datasets.FOREX_EURUSD_1H_ASK),
        'window_size': 60,
        'frame_bound': (60, len(datasets.FOREX_EURUSD_1H_ASK))
    }
)

