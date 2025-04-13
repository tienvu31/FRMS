import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from src.utils.logger import LOGGER
from arch import arch_model
from src.common.consts import CommonConsts
from src.services.strategies.stock_predictions.stock_rnn_strategy import StockRNNStratgy
from src.data.processors import Processors
from src.services.strategies.stock_predictions.models import StockRNN
from src.services.strategies.stock_predictions.trainers import ModelTrainer
from src.services.strategies import (
    PortfolioAutoCorr,
    PortfolioDistance,
    PortfolioGarch,
    PortfolioEDA,
    PortfolioRatios,
    PortfolioSpectralDensity,
    PortfolioStationary,
)

st.set_page_config(page_title="KLTN-FRM Analysis", layout="wide")

st.title('FRM Analysis')

tab1, tab2, tab3 = st.tabs(['Table','Analysis','Prediction'])

# Lưu datasets vào session_state
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('**Upload your datasets here**')
        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    
    with col2:
        if uploaded_files:
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                df = pd.read_csv(uploaded_file, thousands=',')  
                
                if "Date" in df.columns:
                    df["Date"] = df["Date"]  
                # comment Date ở các file analysis 
                st.session_state["datasets"][filename] = df
                st.write(f"**Dataset: {filename}**")
                st.write(df)

                # 1autocorrelation & autocovariance
                analyzer = PortfolioAutoCorr()
                autocorr_df, autocov_df = analyzer.analyze(df)
                st.session_state[f"{filename}_autocorr_df"] = autocorr_df
                st.session_state[f"{filename}_autocov_df"] = autocov_df
                
                # distance
                distance_analyzer = PortfolioDistance()
                mst, linkage_matrix, distance_df = distance_analyzer.analyze(df)
                st.session_state[f"{filename}_distance_df"] = distance_df
                st.session_state[f"{filename}_mst"] = mst
                st.session_state[f"{filename}_linkage_matrix"] = linkage_matrix
                
                # spectral density
                spectral_analyzer = PortfolioSpectralDensity()
                log_returns = spectral_analyzer.analyze(df)
                st.session_state[f"{filename}_spectral_log_returns"] = log_returns


with tab2:
    col1, col2, col3 = st.columns(3)

    # ---- Column 1: Correlation & Autocovariance ----
    with col1:
        with st.container(border=True):
            with st.expander('Correlation Analysis'):
                st.write('Autocorrelation')
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    for filename in st.session_state["datasets"].keys():
                        autocorr_df = st.session_state.get(f"{filename}_autocorr_df")

                        if isinstance(autocorr_df, pd.DataFrame):
                            fig, ax = plt.subplots(figsize=(6, 4))
                            for symbol in autocorr_df.columns:
                                ax.plot(autocorr_df.index, autocorr_df[symbol], label=symbol, marker='o', alpha=0.5, lw=1)
                            ax.set_title(f'Autocorrelation Plot - {filename}')
                            ax.legend()
                            st.pyplot(fig)

                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(autocorr_df.T, cmap="coolwarm", annot=False, ax=ax)
                            ax.set_title(f'Autocorrelation Heatmap - {filename}')
                        st.pyplot(fig)

                st.write('Autocovariance')
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    for filename in st.session_state["datasets"].keys():
                        autocov_df = st.session_state.get(f"{filename}_autocov_df")

                        if isinstance(autocov_df, pd.DataFrame):
                            fig, ax = plt.subplots(figsize=(6, 4))
                            for symbol in autocov_df.columns:
                                ax.plot(autocov_df.index, autocov_df[symbol], label=symbol, marker='o', alpha=0.5, lw=1)
                            ax.set_title(f'Autocovariance Plot - {filename}')
                            ax.legend()
                            st.pyplot(fig)

                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(autocov_df.T, cmap="coolwarm", annot=False, ax=ax)
                            ax.set_title(f'Autocovariance Heatmap - {filename}')
                            st.pyplot(fig)

    # ---- Column 2: Distance Analysis ----
    with col2:
        with st.container(border=True):
            with st.expander('Distance Analysis'):
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    for filename in st.session_state["datasets"].keys():
                        distance_df = st.session_state.get(f"{filename}_distance_df")
                        if isinstance(distance_df, pd.DataFrame):
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(distance_df, annot=True, cmap="coolwarm", fmt=".1f", ax=ax)
                            ax.set_title(f"Distance Heatmap - {filename}")
                            st.pyplot(fig)

                        if mst:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            pos = nx.spring_layout(mst, seed=42)
                            nx.draw(mst, pos, with_labels=True, node_size=2500, font_size=10, 
                                    edge_color="blue", node_color="lightgreen", ax=ax)
                            ax.set_title(f"Minimum Spanning Tree - {filename}")
                            st.pyplot(fig)

                        if isinstance(linkage_matrix, np.ndarray) and isinstance(distance_df, pd.DataFrame):
                            fig, ax = plt.subplots(figsize=(6, 4))
                            dendrogram(linkage_matrix, labels=distance_df.columns, leaf_rotation=90, leaf_font_size=10, ax=ax)
                            ax.set_title(f"Dendrogram - {filename}")
                            st.pyplot(fig)
          
    # ---- Column 3: GARCH ----
    with col3:
        with st.container(border=True):
            with st.expander('GARCH Volatility Analysis'):
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    for filename, df in st.session_state.get("datasets", {}).items():
                        st.write(f"**Dataset: {filename}**")
                        garch_analyzer = PortfolioGarch()
                        returns = garch_analyzer.analyze(df)

                        symbols = returns.columns
                        num_symbols = len(symbols)
                        num_rows = (num_symbols + 1) // 2  # Tính số hàng (2 cột cố định)
                        
                        # Điều chỉnh thành 3 hàng nếu số mã cổ phiếu nhiều hơn 6
                        num_rows = max(3, (num_symbols + 1) // 2)  

                        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 4 * num_rows))
                        axes = axes.flatten()  # Biến mảng 2D thành 1D để dễ indexing

                        for i, symbol in enumerate(symbols):
                            model = arch_model(returns[symbol].dropna(), vol="Garch", p=1, q=1, mean="Constant", dist="Normal")
                            res = model.fit(disp="off")
                            res.conditional_volatility.plot(ax=axes[i], color='red')

                            axes[i].set_title(f"GARCH Volatility - {symbol}", fontsize=12, weight='bold')
                            axes[i].set_ylabel(r'$\sigma_t$', fontsize=10)
                            axes[i].set_xlabel("Time", fontsize=10)
                            axes[i].grid(True)

                        # Xóa subplot thừa nếu có
                        for j in range(i + 1, len(axes)):
                            fig.delaxes(axes[j])

                        plt.tight_layout()
                        st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)  
    with col1:
        with st.container(border=True):
            with st.expander("Exploratory Data Analysis"):
                st.write('Scatter & Regression')
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    for filename, df in st.session_state["datasets"].items():
                        eda_analyzer = PortfolioEDA()
                        eda_df = eda_analyzer.analyze(df)
                        indices = CommonConsts.ticker_model

                        fig, axes = plt.subplots(6, 6, figsize=(24, 20))
                        for i, j in [(i, j) for i in range(len(indices)) for j in range(len(indices))]:
                            ax = axes[i, j]
                            if i != j:
                                ax.scatter(eda_df[indices[i]], eda_df[indices[j]], s=5, alpha=0.2, color='red')
                                sns.regplot(x=eda_df[indices[i]], y=eda_df[indices[j]], ax=ax, scatter=False, color='red', 
                                            line_kws={'color': 'blue', 'linewidth': 2})
                            else:
                                ax.plot(eda_df[indices[i]], color='blue', alpha=0.5)
                            ax.set_xlabel(indices[i])
                            ax.set_ylabel(indices[j])
                            ax.set_title(f"Scatter & Regression - {filename}")
                        st.pyplot(fig)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(eda_df[indices].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
                    ax.set_title(f"Correlation Heatmap - {filename}")
                    st.pyplot(fig)

    with col2:
        with st.container(border = True):
            with st.expander('Spectraldensity'):
                if "datasets" in st.session_state and st.session_state["datasets"]:

                    for filename in st.session_state["datasets"].keys():
                        log_returns = st.session_state.get(f"{filename}_spectral_log_returns")
                        if log_returns is not None:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            spectral_analyzer = PortfolioSpectralDensity()
                            
                            for symbol in log_returns.columns:
                                freqs, psd = spectral_analyzer.compute_spectral_density(log_returns[symbol])
                                ax.plot(freqs, psd, label=symbol)

                            ax.set_title('Spectral Density of Log Returns', fontsize=12, weight='bold')
                            ax.set_xlabel('Frequency', fontsize=12, weight='bold')
                            ax.set_ylabel('Power Spectral Density', fontsize=12, weight='bold')
                            ax.set_yscale('log')
                            ax.legend(ncol=5)
                            ax.grid(True)
                            
                            st.pyplot(fig)
    
        with col3:
            with st.container(border=True):
                with st.expander('Stationarity Analysis'):
                    if "datasets" in st.session_state and st.session_state["datasets"]:

                        for filename, df in st.session_state["datasets"].items():
                            st.write(f"**Dataset: {filename}**")

                            stationary_analyzer = PortfolioStationary()
                            df_stationary = stationary_analyzer.analyze(df)

                            indices = CommonConsts.ticker_model
                            for index in range(len(indices)):
                                symbol = indices[index]
                                prices = df_stationary[symbol]

                                # Tính log return
                                log_returns = np.log(prices / prices.shift(5)).dropna()

                                # ARIMA models
                                ar_model = AutoReg(log_returns, lags=5).fit()
                                ma_model = ARIMA(log_returns, order=(0, 0, 2)).fit()
                                arma_model = ARIMA(log_returns, order=(2, 0, 2)).fit()
                                arima_model = ARIMA(log_returns, order=(2, 0, 2)).fit()

                                # Vẽ biểu đồ
                                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                                axes[0, 0].plot(log_returns, label='Actual')
                                axes[0, 0].plot(ar_model.predict(start=0, end=len(log_returns)-1), label='AR(2)', color='orange', alpha=0.5)
                                axes[0, 0].set_title(f'AR Model - {symbol}', fontsize=12, weight='bold')

                                axes[0, 1].plot(log_returns, label='Actual')
                                axes[0, 1].plot(ma_model.predict(start=0, end=len(log_returns)-1), label='MA(2)', color='green', alpha=0.5)
                                axes[0, 1].set_title(f'MA Model - {symbol}', fontsize=12, weight='bold')

                                axes[1, 0].plot(log_returns, label='Actual')
                                axes[1, 0].plot(arma_model.predict(start=0, end=len(log_returns)-1), label='ARMA(2,2)', color='purple', alpha=0.5)
                                axes[1, 0].set_title(f'ARMA Model - {symbol}', fontsize=12, weight='bold')

                                axes[1, 1].plot(log_returns, label='Actual')
                                axes[1, 1].plot(arima_model.predict(start=0, end=len(log_returns)-1), label='ARIMA(2,1,2)', color='red', alpha=0.5)
                                axes[1, 1].set_title(f'ARIMA Model - {symbol}', fontsize=12, weight='bold')

                                for ax in axes.flatten():
                                    ax.legend()
                                    ax.grid(True)

                                st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)
    with col1:
# ---- Column 3: Ratios Analysis ----

        with st.container(border=True):
            with st.expander('Ratios Analysis'):
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    for filename in st.session_state["datasets"].keys():
                        df = st.session_state["datasets"][filename]
                        
                        # Khởi tạo và phân tích tỷ lệ tài chính
                        analyzer = PortfolioRatios()
                        results_df, weights_df = analyzer.analyze(df)
                        
                        # Trực quan hóa tỷ lệ tài chính
                        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                        
                        # Sharpe Ratio
                        axes[0, 0].scatter(
                            results_df["Volatility"],
                            results_df["Return"],
                            c=results_df["Sharpe Ratio"],
                            cmap="viridis",
                            marker="o",
                            s=5,
                            alpha=0.5,
                        )
                        axes[0, 0].set_title("Sharpe Ratio", fontsize=12, weight="bold")
                        axes[0, 0].set_xlabel("Volatility", fontsize=12, weight="bold")
                        axes[0, 0].set_ylabel("Return", fontsize=12, weight="bold")

                        # Treynor Ratio
                        axes[0, 1].scatter(
                            results_df["Volatility"],
                            results_df["Return"],
                            c=results_df["Treynor Ratio"],
                            cmap="viridis",
                            marker="o",
                            s=5,
                            alpha=0.5,
                        )
                        axes[0, 1].set_title("Treynor Ratio", fontsize=12, weight="bold")
                        axes[0, 1].set_xlabel("Volatility", fontsize=12, weight="bold")
                        axes[0, 1].set_ylabel("Return", fontsize=12, weight="bold")

                        # Jensen's Alpha
                        axes[1, 0].scatter(
                            results_df["Volatility"],
                            results_df["Return"],
                            c=results_df["Jensen's Alpha"],
                            cmap="viridis",
                            marker="o",
                            s=5,
                            alpha=0.5,
                        )
                        axes[1, 0].set_title("Jensen's Alpha", fontsize=12, weight="bold")
                        axes[1, 0].set_xlabel("Volatility", fontsize=12, weight="bold")
                        axes[1, 0].set_ylabel("Return", fontsize=12, weight="bold")

                        # Sortino Ratio
                        axes[1, 1].scatter(
                            results_df["Volatility"],
                            results_df["Return"],
                            c=results_df["Sortino Ratio"],
                            cmap="viridis",
                            marker="o",
                            s=5,
                            alpha=0.5,
                        )
                        axes[1, 1].set_title("Sortino Ratio", fontsize=12, weight="bold")
                        axes[1, 1].set_xlabel("Volatility", fontsize=12, weight="bold")
                        axes[1, 1].set_ylabel("Return", fontsize=12, weight="bold")

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Cập nhật phần weights của danh mục đầu tư
                        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
                        axes = axes.flatten()

                        metrics = ["Sharpe Ratio", "Treynor Ratio", "Jensen's Alpha", "Sortino Ratio"]
                        for i, metric in enumerate(metrics):
                            best_metric = metric
                            best_idx = results_df[metric].idxmax()  # Tìm chỉ số tốt nhất

                            best_portfolio_weights = weights_df.iloc[best_idx]
                            best_portfolio_return = results_df["Return"][best_idx]
                            best_portfolio_volatility = results_df["Volatility"][best_idx]

                            LOGGER.info(f"Best portfolio based on {best_metric}:\n")
                            LOGGER.info(f"\nExpected Annualized Return: {best_portfolio_return * 252:.2%}")
                            LOGGER.info(f"Expected Annualized Volatility: {best_portfolio_volatility * np.sqrt(252):.2%}")

                            axes[i].bar(best_portfolio_weights.index, best_portfolio_weights, color='blue', label = metric, alpha = 0.8)
                            axes[i].set_title(f"{best_metric}", fontsize = 12, weight = 'bold')
                            axes[i].set_xlabel("Asset", fontsize = 12, weight = 'bold')
                            axes[i].set_ylabel("Weight", fontsize = 12, weight = 'bold')
                            axes[i].set_xticklabels(best_portfolio_weights.index, rotation=90 , fontsize = 12, weight = 'bold')

                        plt.tight_layout()
                        st.pyplot(fig)


        # with st.container(border=True):
        #     with st.expander('Ratios Analysis'):
                # if "datasets" in st.session_state and st.session_state["datasets"]:
                #     for filename, df in st.session_state["datasets"].items():
                #         st.write(f"**Portfolio Ratios - {filename}**")
                        
                #         ratios_analyzer = PortfolioRatios()
                #         try:
                #             results_df, weights_df = ratios_analyzer.analyze(df)
                #             indices = results_df.columns

                #             # Plot các chỉ số (4 biểu đồ)
                #             fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                #             axes = axes.flatten()
                #             metrics = ["Sharpe Ratio", "Treynor Ratio", "Jensen's Alpha", "Sortino Ratio"]
                #             for i, metric in enumerate(metrics):
                #                 scatter = axes[i].scatter(
                #                     results_df["Volatility"],
                #                     results_df["Return"],
                #                     c=results_df[metric],
                #                     cmap="viridis",
                #                     s=5,
                #                     alpha=0.5,
                #                 )
                #                 axes[i].set_title(metric)
                #                 axes[i].set_xlabel("Volatility")
                #                 axes[i].set_ylabel("Return")
                #                 fig.colorbar(scatter, ax=axes[i])
                #             st.pyplot(fig)

                #             # Plot portfolio weights tốt nhất
                #             fig, axes = plt.subplots(2, 2, figsize=(12, 6))
                #             axes = axes.flatten()
                #             for i, metric in enumerate(metrics):
                #                 idx = results_df[metric].idxmax()
                #                 best_weights = weights_df.iloc[idx]
                #                 axes[i].bar(best_weights.index, best_weights.values)
                #                 axes[i].set_title(f"Best Portfolio - {metric}")
                #                 axes[i].set_xlabel("Asset")
                #                 axes[i].set_ylabel("Weight")
                #                 axes[i].tick_params(axis='x', rotation=90)
                #             st.pyplot(fig)
                        
                #         except Exception as e:
                #             st.error(f"Error analyzing {filename}: {e}")
                        

# with tab3:
#     st.subheader("📈 Stock Price Prediction using RNN")
#     rnn_strategy = StockRNNStratgy()

#     if "datasets" in st.session_state and st.session_state["datasets"]:
#         for filename, df in st.session_state["datasets"].items():
#             with st.expander(f"Mô hình dự đoán cho: {filename}"):
#                 try:
#                     rnn_strategy.visualize(df)
#                 except Exception as e:
#                     st.error(f"Đã xảy ra lỗi khi xử lý {filename}: {e}")
#     else:
#         st.warning("Vui lòng upload dữ liệu ở tab Table trước khi thực hiện dự đoán.")
