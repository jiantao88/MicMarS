from mlib.core.action import Action
import pandas as pd
from mlib.core.base_agent import BaseAgent
import numpy as np
from typing import Dict, Any
from market_simulation.states.trade_info_state import TradeInfoState
from mlib.core.observation import Observation
import yfinance as yf
import os
from pathlib import Path

class RealDataAgent(BaseAgent):
    def __init__(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        interval: str = "15m"  # 15分钟间隔
    ):
        super().__init__(
            init_cash=1000,
            communication_delay=0,
            computation_delay=0,
        )
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        
        # 检查缓存
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{symbol}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}_{interval}.pkl"
        
        if cache_file.exists():
            self.historical_data = pd.read_pickle(cache_file)
            print(f"从缓存加载数据: {cache_file}")
        else:
            print(f"从 Yahoo Finance 获取数据...")
            # 获取历史数据
            ticker = yf.Ticker(symbol)
            self.historical_data = ticker.history(
                start=start_time.strftime('%Y-%m-%d'),
                end=end_time.strftime('%Y-%m-%d'),
                interval=interval
            )
            if self.historical_data.empty:
                raise ValueError(f"No data found for {symbol} between {start_time} and {end_time}")
            # 保存缓存
            self.historical_data.to_pickle(cache_file)
            print(f"数据已缓存到: {cache_file}")
        
        # 初始化数据索引
        self.current_index = 0
        self.data_times = self.historical_data.index.tolist()

    def get_action(self, observation: Observation) -> Action:
        assert self.agent_id == observation.agent.agent_id
        time = observation.time
        
        if time < self.start_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=self.start_time)
        
        if time > self.end_time or self.current_index >= len(self.data_times):
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=None)

        # 获取当前时间点的数据
        current_data = self.historical_data.iloc[self.current_index]
        
        # 使用实际的开盘价、最高价、最低价和收盘价创建订单
        orders = []
        price = current_data['Close']  # 使用收盘价
        volume = int(current_data['Volume'] / 100)  # 将成交量缩小100倍以适应模拟
        
        # 创建买入或卖出订单（这里简单地根据价格变动决定）
        if self.current_index > 0:
            prev_price = self.historical_data.iloc[self.current_index - 1]['Close']
            order_type = 'B' if price > prev_price else 'S'
        else:
            order_type = 'B'
        
        orders = self.construct_valid_orders(
            time=time,
            symbol=self.symbol,
            type=order_type,
            price=price,
            volume=max(100, min(volume, 2000))  # 限制成交量在100-2000之间
        )
        
        # 更新索引
        self.current_index += 1
        
        # 设置下一次唤醒时间
        next_wakeup_time = time + pd.Timedelta(minutes=15) if self.current_index < len(self.data_times) else None
        
        return Action(agent_id=self.agent_id, orders=orders, time=time, next_wakeup_time=next_wakeup_time)

    def _sample(self, probs: Dict[Any, float]) -> Any:
        """Sample from a discrete probability distribution"""
        choices = list(probs.keys())
        probs = list(probs.values())
        return np.random.choice(choices, p=probs)
