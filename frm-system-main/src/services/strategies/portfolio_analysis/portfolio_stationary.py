import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg


from src.common.consts import CommonConsts
from src.services.strategies.strategy_interface import StrategyInterface
from src.utils.logger import LOGGER


class PortfolioStationary(StrategyInterface):
    def analyze(self, df: pd.DataFrame):
        # df["Date"] = pd.to_datetime(df["Date"])
        # df.set_index("Date", inplace=True)

        return df
    
    def visualize(self, df):
        df = self.analyze(df)
        indices = CommonConsts.ticker_model
        for index in range(len(indices)):
            symbol = indices[index]
            prices = df[symbol]

            # Calculate log returns
            log_returns = np.log(prices / prices.shift(5)).dropna()
            # Check stationarity
            adf_test = adfuller(log_returns)
            LOGGER.info('--------------')
            LOGGER.info(symbol)
            LOGGER.info(f'ADF Statistic: {adf_test[0]}')
            LOGGER.info(f'p-value: {adf_test[1]}')

            log_returns_mean = log_returns.mean()
            log_returns_std = log_returns.std()
            log_returns = (log_returns - log_returns_mean)/(log_returns_std)

            log_returns.index = pd.RangeIndex(start=0, stop=len(log_returns), step=1)
            returns_votality = prices.std()


            # Autoregressive (AR) Model
            ar_model = AutoReg(log_returns, lags=5).fit()
            ar_pred = ar_model.predict(start=0, end=len(log_returns)-1)
            # Moving Average (MA) Model
            ma_model = ARIMA(log_returns, order=(0, 0, 2)).fit()
            ma_pred = ma_model.predict(start=0, end=len(log_returns)-1)
            # Autoregressive Moving Average (ARMA)
            arma_model = ARIMA(log_returns, order=(2, 0, 2)).fit()
            arma_pred = arma_model.predict(start=0, end=len(log_returns)-1)
            # Autoregressive Integrated Moving Average (ARIMA)
            arima_model = ARIMA(log_returns, order=(2, 0, 2)).fit()  # Adjust p, d, q based on analysis
            arima_pred = arima_model.predict(start=0, end=len(log_returns)-1)
            # Displaying fitted parameters for each model
            LOGGER.info("Autoregressive (AR) Model Parameters:")
            LOGGER.info(ar_model.params)

            LOGGER.info("\nMoving Average (MA) Model Parameters:")
            LOGGER.info(ma_model.params)

            LOGGER.info("\nAutoregressive Moving Average (ARMA) Model Parameters:")
            LOGGER.info(arma_model.params)

            LOGGER.info("\nAutoregressive Integrated Moving Average (ARIMA) Model Parameters:")
            LOGGER.info(arima_model.params)

            fig, axes = plt.subplots(2, 2, figsize=(16, 6))

            axes[0, 0].plot(log_returns, label='Actual')
            axes[0, 0].plot(ar_pred, label='AR(2)', color='orange', marker = 'o', alpha = 0.2)
            axes[0, 0].set_title('AR Model', fontsize = 12, weight = 'bold')

            axes[0, 1].plot(log_returns, label='Actual')
            axes[0, 1].plot(ma_pred, label='MA(2)', color='green', marker = 'o', alpha = 0.2)
            axes[0, 1].set_title('MA Model', fontsize = 12, weight = 'bold')

            axes[1, 0].plot(log_returns, label='Actual')
            axes[1, 0].plot(arma_pred, label='ARMA(2,2)', color='purple', marker = 'o', alpha = 0.2)
            axes[1, 0].set_title('ARMA Model', fontsize = 12, weight = 'bold')

            axes[1, 1].plot(log_returns, label='Actual')
            axes[1, 1].plot(arima_pred, label='ARIMA(2,1,2)', color='red', marker = 'o', alpha = 0.2)
            axes[1, 1].set_title('ARIMA Model', fontsize = 12, weight = 'bold')


            for ax in axes.flatten():
                ax.legend(loc='lower right')
                ax.set_xlabel('Time [day]', fontsize = 12, weight = 'bold')
                ax.set_ylabel('Log Return', fontsize = 12, weight = 'bold')
                ax.grid(True)
            plt.tight_layout()
            plt.savefig(f'{CommonConsts.IMG_FOLDER}\\stationary_figures\\stationary_{symbol}.jpg', dpi = 600)