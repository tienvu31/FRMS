import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

from src.common.consts import CommonConsts
from src.services.strategies.strategy_interface import StrategyInterface

class PortfolioGarch(StrategyInterface):
    def analyze(self, df: pd.DataFrame):
        indices = CommonConsts.ticker_model
        # df['Date'] = pd.to_datetime(df['Date'])
        # df.set_index('Date', inplace=True)
        returns = df[indices].pct_change().dropna()

        return returns


    def visualize(self, df):
        vol_model = 'Garch'
        indices = CommonConsts.ticker_model
        returns = self.analyze(df)

        num_symbols = len(indices)
        fig, axes = plt.subplots(nrows=(num_symbols + 1) // 2, ncols=2, figsize=(12, 10))
        axes = axes.flatten()

        results = {}
        for i, symbol in enumerate(indices):
            model = arch_model(returns[symbol], vol=vol_model, p=1, q=1, 
                mean='Constant', dist='Normal')
            res = model.fit(disp='off')
            results[symbol] = res

            res.conditional_volatility.plot(ax=axes[i], color = 'red')
            axes[i].set_ylabel(r'$\sigma \times 10^3$', fontsize = 14, weight = 'bold')
            axes[i].set_title(label=f'{symbol}', fontsize=16, weight = 'bold')
            axes[i].grid(True)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\garch.jpg', dpi = 600)

