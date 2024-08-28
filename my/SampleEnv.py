import pandas as pd
from tensortrade.feed.core import DataFeed, Stream, NameSpace
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.actions import BSH, ManagedRiskOrders
from tensortrade.env.default.rewards import PBR, SimpleProfit
import tensortrade.env.default as default

# import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))

USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 8, "TensorTrade Coin")

def create_env():
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
    df.drop(columns=['datetime'], inplace=True)

    close_price = Stream.source(list(price_history['close']), dtype="float").rename("USD-TTC")
    exchange = Exchange("exchange", service=execute_order)(
        close_price
    )

    portfolio = Portfolio(USD, [
        Wallet(exchange, 10000 * USD),
        Wallet(exchange, 0 * TTC),
    ])

    streams = []
    with NameSpace("exchange"):
        for name in df.columns:
            streams += [Stream.source(list(df[name]), dtype="float").rename(name)]

    feed = DataFeed(streams)

    action_scheme = BSH(cash=Wallet(exchange, 10000 * USD), asset=Wallet(exchange, 0 * TTC))
    reward_scheme = PBR(price=close_price)

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=20,
        enable_logger=False,
    )

    return env


env = create_env()

terminated = False
obs = env.reset()
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

print(f"net = {env.action_scheme.portfolio.net_worth}")
print(f"p/l {env.action_scheme.portfolio.profit_loss}")
print("end.")