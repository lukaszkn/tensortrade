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

# import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))

USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 2, "TensorTrade Coin")

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

    df = load_csv('UpDown1.csv')
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

        streams += [Stream.source(list(df['close']), dtype="float").rolling(window=10).mean().rename("fast")]
        streams += [Stream.source(list(df['close']), dtype="float").rolling(window=50).mean().rename("medium")]

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
    .resources(num_gpus=1)
)
algo = config.build()

for i in range(300):
    results = algo.train()
    print("Iter: {0}; ep_mean= {1:.0f}  ep_min= {2:.0f}  ep_max= {3:.0f}".format(i, results['episode_reward_mean'], results['episode_return_min'], results['episode_return_max']))

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
print(f"p/l {env.action_scheme.portfolio.profit_loss}")

env.render()

print("end.")