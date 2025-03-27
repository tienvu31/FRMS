import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

from src.common.consts import CommonConsts
from src.services.strategies.strategy_interface import StrategyInterface


class PortfolioDistance(StrategyInterface):
    def analyze(self, df: pd.DataFrame):
        df['Date'] = pd.to_datetime(df['Date'])  
        df.set_index('Date', inplace=True)

        lag = 5
        log_DF = pd.DataFrame([])
        for index in range(len(df.columns)):
            symbol = df.columns[index]
            prices = df[symbol]
            log_returns = np.log(prices / prices.shift(lag)).dropna()
            log_returns_mean = log_returns.mean()
            log_returns_std = log_returns.std()
            log_returns = (log_returns - log_returns_mean)/(log_returns_std)
            log_DF[symbol] = log_returns

        standardized_df = (log_DF - log_DF.mean()) / log_DF.std()
        distance_matrix = squareform(pdist(standardized_df.T, metric='euclidean'))
        distance_df = pd.DataFrame(distance_matrix, index=df.columns, columns=df.columns)

        mst_graph = nx.Graph()

        for i, stock1 in enumerate(distance_df.columns):
            for j, stock2 in enumerate(distance_df.columns):
                if i < j:
                    mst_graph.add_edge(stock1, stock2, weight=df.iloc[i, j])

        mst = nx.minimum_spanning_tree(mst_graph, algorithm="kruskal")
        
        # Step 2: Generate hierarchical clustering tree
        linkage_matrix = linkage(squareform(distance_matrix), method="ward")

        return mst, linkage_matrix, distance_df

    def visualize(self, df):    
        mst, linkage_matrix, distance_df = self.analyze(df)

        # Plot pairwise distance matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(distance_df, annot=True, cmap="coolwarm", fmt=".1f")
        plt.xticks(fontsize=16, weight='bold', rotation=90)
        plt.yticks(fontsize=16, weight='bold', rotation=0)
        plt.tight_layout()
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\distance_01.jpg', dpi=600)

        # Plot MST
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(mst, seed=42)
        nx.draw(mst, pos, with_labels=True, node_size=2500, font_size=16, edge_color="blue", node_color="lightgreen")
        plt.title("Minimum Spanning Tree", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\distance_02.jpg', dpi=600)

        # Plot hierarchical clustering tree
        plt.figure(figsize=(6, 6))
        dendrogram(linkage_matrix, labels=distance_df.columns, leaf_rotation=90, leaf_font_size=10)
        plt.xlabel("Stocks", fontsize = 16, weight = 'bold')
        plt.ylabel("Distance", fontsize = 16, weight = 'bold')
        plt.xticks(fontsize = 16, weight = 'bold')
        plt.yticks(fontsize = 16, weight = 'bold')
        plt.tight_layout()
        plt.savefig(f'{CommonConsts.IMG_FOLDER}\\distance_03.jpg', dpi=600)