from mlib.core.action import Action
import pandas as pd
import numpy as np
from mlib.core.base_agent import BaseAgent
from mlib.core.observation import Observation
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from typing import Tuple, List
import logging

class PredictionAgent(BaseAgent):
    def __init__(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        lookback_days: int = 60,  # 使用过去60天的数据来训练
        prediction_days: int = 7,  # 预测未来7天
        feature_columns: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume']
    ):
        super().__init__(
            init_cash=1000,
            communication_delay=0,
            computation_delay=0,
        )
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.feature_columns = feature_columns
        self.model = None
        self.scaler = MinMaxScaler()
        
        # 获取训练数据
        self.historical_data = self.get_historical_data()
        if self.historical_data is None:
            raise ValueError("获取历史数据失败")
        
        # 创建并训练模型
        self._create_and_train_model()
        
    def get_historical_data(self):
        """获取历史数据"""
        try:
            data = yf.download(
                self.symbol,
                start=self.start_time.strftime('%Y-%m-%d'),
                end=self.end_time.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if len(data) < self.prediction_days:
                logging.warning(f"获取到的数据点数({len(data)})少于所需的预测天数({self.prediction_days})")
                return None
                
            logging.info(f"获取到 {len(data)} 天的历史数据")
            return data
            
        except Exception as e:
            logging.error(f"获取历史数据时发生错误: {str(e)}")
            return None

    def prepare_features(self, data):
        """准备特征数据"""
        df = data.copy()
        close_prices = df['Close']
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = close_prices.ewm(span=12, adjust=False).mean()
        exp2 = close_prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        ma20 = close_prices.rolling(window=20).mean()
        rolling_std = close_prices.rolling(window=20).std()
        bb_upper = ma20 + (2 * rolling_std)
        bb_lower = ma20 - (2 * rolling_std)
        
        # EMA
        ema12 = close_prices.ewm(span=12, adjust=False).mean()
        ema26 = close_prices.ewm(span=26, adjust=False).mean()
        
        # 创建特征DataFrame
        features = pd.DataFrame(index=df.index)
        features['Close'] = close_prices
        features['RSI'] = rsi
        features['MACD'] = macd
        features['MACD_Hist'] = macd - signal_line
        features['BB_Position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
        features['EMA_Ratio'] = ema12 / ema26
        features['Price_Change'] = close_prices.pct_change()
        
        # 处理NaN值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # 标准化数据
        features = pd.DataFrame(
            self.scaler.fit_transform(features), 
            columns=features.columns, 
            index=features.index
        )
        
        return features

    def _prepare_data(self, data):
        """准备训练数据"""
        features = self.prepare_features(data)
        
        # 创建序列数据
        X, y = [], []
        for i in range(len(features) - self.prediction_days):
            X.append(features.iloc[i:(i + self.prediction_days)].values)
            y.append(features.iloc[i + self.prediction_days]['Close'])
            
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """构建增强的LSTM模型"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _create_and_train_model(self):
        """创建并训练LSTM模型"""
        if len(self.historical_data) < self.lookback_days:
            raise ValueError("没有足够的历史数据进行训练")
            
        # 准备训练数据
        X, y = self._prepare_data(self.historical_data)
        
        # 创建LSTM模型
        self.model = self.build_model((self.prediction_days, X.shape[2]))
        
        # 训练模型
        self.model.fit(X, y, epochs=25, batch_size=32, verbose=1)
        logging.info("模型训练完成")
        
    def predict_next_days(self, days=3):
        """预测未来几天的价格"""
        if not hasattr(self, 'model'):
            self._create_and_train_model()
            
        # 准备特征数据
        features = self.prepare_features(self.historical_data)
        sequence_length = self.prediction_days
        
        # 获取最新序列
        last_sequence = features.iloc[-sequence_length:].values.reshape(1, sequence_length, features.shape[1])
        
        # 预测未来价格
        predictions = []
        confidence_levels = [0.95, 0.825, 0.7]  # 随时间递减的置信度
        
        current_sequence = last_sequence.copy()
        for _ in range(days):
            # 预测下一天
            pred = self.model.predict(current_sequence, verbose=0)
            scaled_pred = pred[0][0]
            
            # 反向转换预测价格
            original_close_prices = self.historical_data['Close'].values.reshape(-1, 1)
            price_scaler = MinMaxScaler()
            price_scaler.fit(original_close_prices)
            actual_price = price_scaler.inverse_transform([[scaled_pred]])[0][0]
            
            predictions.append(actual_price)
            
            # 更新序列用于下一次预测
            new_row = current_sequence[0, 1:].copy()
            new_features = features.iloc[-1:].values.copy()
            new_features[0, features.columns.get_loc('Close')] = scaled_pred
            current_sequence = np.vstack([new_row, new_features]).reshape(1, sequence_length, features.shape[1])
        
        # 生成预测日期
        last_date = self.historical_data.index[-1]
        prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        # 创建预测结果DataFrame
        predictions_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Price': predictions,
            'Confidence': confidence_levels[:len(predictions)]
        })
        
        print("\n未来{}天的价格预测：".format(days))
        print(predictions_df)
        
        return predictions_df

    def get_action(self, observation: Observation) -> Action:
        """根据观察结果决定行动"""
        if not hasattr(self, 'last_prediction_time'):
            self.last_prediction_time = None
            
        current_time = observation.time
        
        # 只在每天开始时进行一次预测
        if self.last_prediction_time is None or current_time.date() != self.last_prediction_time.date():
            predictions = self.predict_next_days()
            self.last_prediction_time = current_time
            
            if predictions is not None and len(self.historical_data) > 0:
                # 获取当前价格和预测价格
                current_price = float(self.historical_data['Close'].iloc[-1])
                next_day_price = float(predictions.iloc[0]['Predicted_Price'])
                
                # 根据预测结果决定行动
                if next_day_price > current_price * 1.02:  # 预测价格上涨超过2%
                    return Action(
                        agent_id=self.agent_id,
                        orders=self.construct_valid_orders(
                            time=current_time,
                            symbol=self.symbol,
                            type='B',
                            price=current_price,
                            volume=100  # 固定交易量
                        ),
                        time=current_time,
                        next_wakeup_time=current_time + pd.Timedelta(days=1)
                    )
                elif next_day_price < current_price * 0.98:  # 预测价格下跌超过2%
                    return Action(
                        agent_id=self.agent_id,
                        orders=self.construct_valid_orders(
                            time=current_time,
                            symbol=self.symbol,
                            type='S',
                            price=current_price,
                            volume=100  # 固定交易量
                        ),
                        time=current_time,
                        next_wakeup_time=current_time + pd.Timedelta(days=1)
                    )
        
        return Action(
            agent_id=self.agent_id,
            orders=[],
            time=current_time,
            next_wakeup_time=current_time + pd.Timedelta(days=1)
        )
