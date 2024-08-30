# C:\Projects\lukaszkn\tensortrade\venv\Scripts\tensorboard.exe --bind_all --logdir C:\Users\lynnx\ray_results\PPO_TradingEnv_2024-08-29_18-35-13vj6_y9lh

import pandas as pd
from tensortrade.feed.core import DataFeed, Stream, NameSpace
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.actions import BSH, ManagedRiskOrders, SimpleOrders
from tensortrade.env.default.rewards import PBR, SimpleProfit, RiskAdjustedReturns
import tensortrade.env.default as default
from tensortrade.env.default.renderers import PlotlyTradingChart
import GPUtil
import torch, os, ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Optional, Dict, Union
from ray.rllib.policy import Policy
from ray.rllib import BaseEnv
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.core.rl_module.rl_module import RLModule
import gymnasium as gym
from ray.rllib.utils.typing import EpisodeType, PolicyID
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

# import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))

USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 4, "TensorTrade Coin")

def create_env(env_config):
    def load_csv(filename):
        df = pd.read_csv('data/' + filename, skiprows=0)
        # df.drop(columns=['symbol', 'volume_btc'], inplace=True)

        # Convert the date column type from string to datetime for proper sorting.
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Make sure historical prices are sorted chronologically, oldest first.
        # df.sort_values(by='date', ascending=True, inplace=True)

        # df.reset_index(drop=True, inplace=True)

        # Format timestamps as you want them to appear on the chart buy/sell marks.
        # df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # PlotlyTradingChart expects 'datetime' as the timestamps column name.
        # df.rename(columns={'date': 'datetime'}, inplace=True)

        return df

    df = load_csv('DE40_tf30.csv')
    price_history = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]  # chart data
    price_history.rename(columns={'datetime': 'date'}, inplace=True)
    df.drop(columns=['datetime'], inplace=True)

    close_price = Stream.source(list(price_history['close']), dtype="float").rename("USD-TTC")
    exchange = Exchange("exchange", service=execute_order)(
        close_price
    )

    cash = Wallet(exchange, 100000 * USD)
    asset = Wallet(exchange, 0 * TTC)
    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    streams = []
    with NameSpace("exchange"):
        for name in df.columns:
            streams += [Stream.source(list(df[name]), dtype="float").rename(name)]

    feed = DataFeed(streams)

    reward_scheme = PBR(price=close_price)
    action_scheme = BSH(
        cash=cash,
        asset=asset
    ).attach(reward_scheme)

    chart_renderer = PlotlyTradingChart(
        display=True,  # show the chart on screen (default)
        height=None,  # affects both displayed and saved file height. None for 100% height.
        save_format='html',  # save the chart to an HTML file
        auto_open_html=True  # open the saved HTML chart in a new browser tab
    )

    renderer_feed = DataFeed(
        [Stream.source(price_history[c].tolist(), dtype="float").rename(c) for c in price_history]
    )

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=20,
        enable_logger=False,
        renderer_feed=renderer_feed,
        renderer=chart_renderer
    )

    return env


class MyCallbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        episode.custom_metrics["net_worth"] = episode.worker.env.action_scheme.portfolio.net_worth - episode.worker.env.action_scheme.portfolio.initial_net_worth

register_env("TradingEnv", create_env)

gpus = GPUtil.getGPUs()
print("Num GPUs Available:", len(gpus))
print("torch gpu: ", torch.cuda.device_count())

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
ray.init(num_cpus=4, num_gpus=1, local_mode=False, log_to_driver=True, include_dashboard=False)

config = (
    PPOConfig()
    .environment(
        env="TradingEnv"
    )
    .env_runners(num_env_runners=3)
    .callbacks(MyCallbacks)
    .resources(num_gpus=1)
)
algo = config.build()

for i in range(10000):
    results = algo.train()
    print("Iter: {0}; ep_mean= {1:.0f}  ep_min= {2:.0f}  ep_max= {3:.0f}  net= {4:.0f}".format(i, results['episode_reward_mean'], results['episode_return_min'], results['episode_return_max'],
                                                                                               results['custom_metrics']['net_worth_mean']))

env = create_env(None)
terminated = truncated = False
obs, info = env.reset()
total_reward = 0.0
while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print(f"Played 1 episode; total-reward={total_reward}")
print(f"net = {env.action_scheme.portfolio.net_worth}")

env.render()

print("end.")