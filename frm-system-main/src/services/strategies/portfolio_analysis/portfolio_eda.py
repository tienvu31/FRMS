import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from src.common.consts import CommonConsts
from src.services.strategies.strategy_interface import StrategyInterface

class PortfolioEDA(StrategyInterface):
    def analyze(self, df: pd.DataFrame):
        return df

    def visualize(self, df):
        # indices are columns of the data frame starting from the second column
        indices = CommonConsts.ticker_model
        fig, axes = plt.subplots(5, 5, figsize=(24, 20))
        for i in range(len(indices)):
            for j in range(len(indices)):
                if i != j:
                    ax = axes[i, j]
                    ax.scatter(x=df[indices[i]], y=df[indices[j]], s=5, alpha=0.2, color='red')
                    sns.regplot(
                        x=df[indices[i]],
                        y=df[indices[j]],
                        ax=ax,
                        scatter=False,
                        color='red',
                        line_kws={'color': 'blue', 'linewidth': 2}
                    )
                    ax.set_xlabel(indices[i], fontsize=16, weight='bold')
                    ax.set_ylabel(indices[j], fontsize=16, weight='bold')
                else:
                    axes[i, j].plot(df[indices[i]], color='blue', alpha=0.5)
                    axes[i, j].set_ylabel(indices[i], fontsize=15, weight='bold', color='blue')

        plt.tight_layout()
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\0_correlation.jpg', dpi = 600)

        # Add correlation heatmap
        correlation_matrix = df[indices].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            xticklabels=indices,
            yticklabels=indices
        )
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\1_correlation_heatmap.jpg', dpi = 600)