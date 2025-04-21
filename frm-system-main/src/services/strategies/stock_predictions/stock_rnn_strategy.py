import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st  # <--- Thêm để hiển thị trực tiếp

from src.common.consts import CommonConsts
from src.data.processors import Processors
from src.services.strategies.stock_predictions.models import StockRNN
from src.services.strategies.stock_predictions.trainers import ModelTrainer
from src.services.strategies.strategy_interface import StrategyInterface
from src.utils.logger import LOGGER


class StockRNNStratgy(StrategyInterface):
    def analyze(self, df: pd.DataFrame):
        return df

    # def visualize(self, df: pd.DataFrame):
    #     indices = CommonConsts.ticker_model
    #     df = df.reset_index(drop=True)

    #     for symbol in indices:
    #         if symbol not in df.columns:
    #             st.warning(f"Symbol `{symbol}` không tồn tại trong dataset.")
    #             continue

    #         symbol_df = df[symbol]
    #         data_dict = Processors.prepare_data(symbol_df=symbol_df)
    #         data_dict = Processors.prepare_data(symbol_df=symbol_df)
    #         if data_dict is None:
    #             LOGGER.warning(f"Không thể train/predict cho {symbol}. Bỏ qua.")
    #             continue


    #         model = StockRNN(
    #             input_size=1,
    #             hidden_size=CommonConsts.HIDDEN_SIZE,
    #             num_layers=CommonConsts.NUM_LAYERS,
    #             output_size=1
    #         )

    #         trainer = ModelTrainer(
    #             model=model,
    #             criterion=nn.MSELoss(),
    #             optimizer=torch.optim.Adam(model.parameters(), lr=CommonConsts.LEARNING_RATE)
    #         )

    #         LOGGER.info(f"\nTraining model for {symbol}")
    #         trainer.train_model(train_loader=data_dict['train_loader'])

    #         predictions = trainer.predict(X_test=data_dict['X_test'])
    #         y_test_np = data_dict['y_test'].numpy()

    #         scaler = data_dict['scaler']
    #         predictions = scaler.inverse_transform(predictions)
    #         y_test_np = scaler.inverse_transform(y_test_np)

    #         last_sequence = data_dict['X_test'][-2].reshape(1, CommonConsts.SEQUENCE_LENGTH, 1)
    #         future_predictions = trainer.predict_future_price(last_sequence)
    #         future_predictions = scaler.inverse_transform(future_predictions)

    #         self.plot_prediction(
    #             y_test=y_test_np,
    #             predictions=predictions,
    #             future_predictions=future_predictions,
    #             symbol=symbol
    #         )
    
    def visualize(self, df: pd.DataFrame):
        indices = CommonConsts.ticker_model
        df = df.reset_index(drop=True)

        for symbol in indices:
            if symbol not in df.columns:
                st.warning(f"Symbol `{symbol}` không tồn tại trong dataset.")
                continue

            symbol_df = df[symbol]
            data_dict = Processors.prepare_data(symbol_df=symbol_df)
            if data_dict is None:
                LOGGER.warning(f"Không thể train/predict cho {symbol}. Bỏ qua.")
                continue

            model = StockRNN(
                input_size=1,
                hidden_size=CommonConsts.HIDDEN_SIZE,
                num_layers=CommonConsts.NUM_LAYERS,
                output_size=1
            )

            trainer = ModelTrainer(
                model=model,
                criterion=nn.MSELoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=CommonConsts.LEARNING_RATE)
            )

            LOGGER.info(f"\nTraining model for {symbol}")
            trainer.train_model(train_loader=data_dict['train_loader'])

            predictions = trainer.predict(X_test=data_dict['X_test'])
            y_test_np = data_dict['y_test'].numpy()

            scaler = data_dict['scaler']
            predictions = scaler.inverse_transform(predictions)
            y_test_np = scaler.inverse_transform(y_test_np)

            last_sequence = data_dict['X_test'][-2].reshape(1, CommonConsts.SEQUENCE_LENGTH, 1)

            # Dự đoán 3 tháng (63 ngày)
            future_predictions_3_month = trainer.predict_future_price(last_sequence, days_ahead=CommonConsts.FORECAST_DAYS)
            future_predictions_3_month = scaler.inverse_transform(future_predictions_3_month)

            # Dự đoán 1 năm (252 ngày)
            future_predictions_1_year = trainer.predict_future_price(last_sequence, days_ahead=CommonConsts.FORECAST_YEAR_DAYS)
            future_predictions_1_year = scaler.inverse_transform(future_predictions_1_year)


            # Gọi hàm hiển thị
            self.plot_prediction(
                y_test=y_test_np,
                predictions=predictions,
                future_predictions_3_month=future_predictions_3_month,
                future_predictions_1_year=future_predictions_1_year,  
                symbol=symbol
            )



    def plot_prediction(
        self,
        y_test: np.ndarray,
        predictions: np.ndarray,
        future_predictions_3_month: np.ndarray,
        future_predictions_1_year: np.ndarray,
        symbol: str
    ):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Test set
        axes[0].plot(y_test, label='Actual Prices', color='blue')
        axes[0].plot(predictions, label='Predicted Prices', marker='o', alpha=0.5, color='red')
        axes[0].legend()
        axes[0].set_title(f'{symbol} Price Prediction (Test Set)', weight='bold')
        axes[0].set_xlabel('Time [days]')
        axes[0].set_ylabel('Value [VND]')
        axes[0].grid(True)

        # 3-month forecast
        axes[1].plot(future_predictions_3_month, label='3-Month Forecast', color='green')
        axes[1].legend()
        axes[1].set_title(f'{symbol} 3-Month Forecast', weight='bold')
        axes[1].set_xlabel('Time [days]')
        axes[1].set_ylabel('Value [VND]')
        axes[1].grid(True)

        # 1-week forecast
        axes[2].plot(future_predictions_1_year, label='1-Year Forecast', color='orange')
        axes[2].legend()
        axes[2].set_title(f'{symbol} 1-Year Price Forecast', weight='bold')
        axes[2].set_xlabel('Time [days]')
        axes[2].set_ylabel('Value [VND]')
        axes[2].grid(True)

        plt.tight_layout()
        st.pyplot(fig)
