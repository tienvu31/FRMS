import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch


from src.common.consts import CommonConsts
from src.services.strategies.strategy_interface import StrategyInterface

class PortfolioSpectralDensity(StrategyInterface):
    def analyze(self, df: pd.DataFrame):
        indices = CommonConsts.ticker_model
        log_returns = np.log(df[indices] / df[indices].shift(1)).dropna()
        return log_returns

    def visualize(self, df):
        indices = CommonConsts.ticker_model
        log_returns = self.analyze(df)
        
        plt.figure(figsize=(10, 4))
        for symbol in indices:
            freqs, psd = self.compute_spectral_density(log_returns[symbol])
            plt.plot(freqs, psd, label=symbol)

        plt.title('Spectral Density of Log Returns', fontsize = 12, weight = 'bold')
        plt.xlabel('Frequency', fontsize = 12, weight = 'bold')
        plt.ylabel('Power Spectral Density', fontsize = 12, weight = 'bold')
        plt.yscale('log')  # Log scale for better visualization of spectral features
        plt.legend(ncol = 5)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\spectral_density.jpg', dpi = 600)


    def compute_spectral_density(self, data, fs=1.0):
        freqs, psd = welch(data, fs=fs, nperseg=min(256, len(data)))  # Welch's method
        return freqs, psd