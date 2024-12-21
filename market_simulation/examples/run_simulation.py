from market_simulation.agents.prediction_agent import PredictionAgent
from matplotlib import dates
import matplotlib.pyplot as plt
from mlib.core.trade_info import TradeInfo
from typing import List
from pandas import Timestamp
from market_simulation.states.trade_info_state import TradeInfoState
from mlib.core.env import Env
from mlib.core.exchange import Exchange
from mlib.core.event import create_exchange_events
from mlib.core.exchange_config import create_exchange_config_without_call_auction
from pathlib import Path
import pandas as pd
import seaborn as sns


def run_simulation():
    """Run simulation with Bitcoin-USD prediction."""
    symbols = ["BTC-USD"]  # 使用比特币-美元对
    # 使用过去的真实数据进行训练和预测
    current_date = Timestamp("2024-12-21")  # 当前日期
    start_time = current_date - pd.Timedelta(days=30)  # 从30天前开始
    end_time = current_date    # 到今天结束

    exchange_config = create_exchange_config_without_call_auction(
        market_open=start_time,
        market_close=end_time,
        symbols=symbols,
    )
    exchange = Exchange(exchange_config)

    # 创建预测代理
    agent = PredictionAgent(
        symbol=symbols[0],
        start_time=start_time,
        end_time=end_time,
        lookback_days=20,    # 使用20天的数据进行训练
        prediction_days=3     # 预测未来3天
    )

    exchange.register_state(TradeInfoState())
    env = Env(exchange=exchange, description="Bitcoin-USD prediction")
    env.register_agent(agent)
    env.push_events(create_exchange_events(exchange_config))

    # 运行模拟
    for observation in env.env():
        action = observation.agent.get_action(observation)
        env.step(action)

    # 获取交易信息和预测
    trade_infos: List[TradeInfo] = get_trade_infos(exchange, symbols[0], start_time, end_time)
    predictions = agent.predict_next_days(days=3)

    print(f"获取到 {len(trade_infos)} 个交易信息点")
    print("\n未来3天的价格预测：")
    print(predictions)

    # 绘制价格曲线和预测
    plot_price_curves(trade_infos, predictions, Path("tmp/price_prediction.png"))


def get_trade_infos(exchange: Exchange, symbol: str, start_time: Timestamp, end_time: Timestamp):
    """获取交易信息"""
    state = exchange.states()[symbol][TradeInfoState.__name__]
    assert isinstance(state, TradeInfoState)
    trade_infos = state.trade_infos
    trade_infos = [x for x in trade_infos if start_time <= x.order.time <= end_time]
    return trade_infos


def plot_price_curves(trade_infos: List[TradeInfo], predictions: pd.DataFrame, path: Path):
    """绘制价格曲线和预测"""
    path.parent.mkdir(parents=True, exist_ok=True)

    # 处理历史数据
    prices = []
    for info in trade_infos:
        if hasattr(info, 'lob_snapshot') and info.lob_snapshot and hasattr(info.lob_snapshot, 'last_price'):
            if info.lob_snapshot.last_price > 0:
                prices.append({
                    "Time": info.order.time,
                    "Price": info.lob_snapshot.last_price,
                    "Type": "Historical"
                })
    
    if not prices:
        print("警告：没有有效的历史价格数据")
        return
        
    prices_df = pd.DataFrame(prices)

    # 添加预测数据
    predictions_data = [
        {
            "Time": row['Date'],
            "Price": row['Predicted_Price'],
            "Type": "Predicted"
        }
        for _, row in predictions.iterrows()
    ]
    predictions_df = pd.DataFrame(predictions_data)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制历史价格
    historical_mask = prices_df['Type'] == 'Historical'
    ax.plot(
        prices_df.loc[historical_mask, 'Time'],
        prices_df.loc[historical_mask, 'Price'],
        label="Historical Price",
        color='blue'
    )

    # 绘制预测价格
    ax.plot(
        predictions_df['Time'],
        predictions_df['Price'],
        label="Predicted Price",
        color='red',
        linestyle="--"
    )

    # 添加置信区间
    if len(prices_df) > 0:
        last_historical_price = prices_df.loc[historical_mask, 'Price'].iloc[-1]
        confidence_range = last_historical_price * 0.1  # 10%的置信区间
        plt.fill_between(
            predictions_df['Time'],
            predictions_df['Price'] - confidence_range,
            predictions_df['Price'] + confidence_range,
            alpha=0.2,
            label="Prediction Confidence Interval"
        )

    ax.set_title("Bitcoin-USD Price - Historical and Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    plt.xticks(rotation=45)
    plt.legend()

    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)
    print(f"保存预测图表到 {path}")


if __name__ == "__main__":
    run_simulation()
