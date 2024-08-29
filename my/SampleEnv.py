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

# import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))

USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 2, "TensorTrade Coin")

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


env = create_env()

terminated = False
obs = env.reset()
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

print(f"net = {env.action_scheme.portfolio.net_worth}")
print(f"p/l {env.action_scheme.portfolio.profit_loss}")

env.render()

print("end.")