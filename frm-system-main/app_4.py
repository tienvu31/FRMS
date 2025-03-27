import streamlit as st 
import pandas as pd
from io import StringIO
st.set_page_config(page_title="KLTN-FRM Analysis", layout="wide")
from src.services.strategies import (
    PortfolioEDA,
    PortfolioAutoCorr,
    PortfolioSpectralDensity,
    PortfolioDistance,
    PortfolioGarch,
    PortfolioRatios,
    PortfolioStationary
)
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from arch import arch_model


st.title('FRM Analysis')

tab1, tab2, tab3 = st.tabs(['Table','Analysis','Prediction'])

# with tab1:
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write('**Upload your dataset here**')
#         with st.expander('Upload Dataset'):
#             uploaded_file = st.file_uploader("Choose a file")
#     with col2:
#         if uploaded_file is not None:
#             # Can be used wherever a "file-like" object is accepted:
#             df = pd.read_csv(uploaded_file, thousands=',')  
#             # df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce').dt.date
#             # df.set_index('Date', inplace=True)  # Đặt cột Date làm index
#             st.write(df)  
                
#             analyzer = PortfolioAutoCorr()
#             autocorr_df, autocov_df = analyzer.analyze(df)
#             st.session_state["autocorr_df"] = autocorr_df
#             st.session_state["autocov_df"] = autocov_df
            
#             distance_analyzer = PortfolioDistance()
#             mst, linkage_matrix, distance_df = distance_analyzer.analyze(df)
#             st.session_state["distance_df"] = distance_df
#             st.session_state["mst"] = mst
#             st.session_state["linkage_matrix"] = linkage_matrix

with tab1:
    st.write("**Upload your dataset here**")
    uploaded_files = st.file_uploader("Chọn các file CSV", accept_multiple_files=True, type=["csv"])

    if uploaded_files:
        st.write(f"Đã tải lên {len(uploaded_files)} file(s)")

        dataframes = {}  # Dictionary lưu các DataFrame

        for file in uploaded_files:
            try:
                df = pd.read_csv(file, thousands=',')  # Giữ nguyên định dạng cột Date
                dataframes[file.name] = df  # Lưu vào dictionary với key là tên file
            except Exception as e:
                st.error(f"Lỗi khi đọc file {file.name}: {e}")

        # Lưu vào session_state để dùng ở tab khác
        st.session_state["datasets"] = dataframes

        # Hiển thị từng DataFrame
        for filename, df in dataframes.items():
            st.write(f"**{filename}**")
            st.write(df)

            # Thực hiện phân tích
            analyzer = PortfolioAutoCorr()
            autocorr_df, autocov_df = analyzer.analyze(df)

            distance_analyzer = PortfolioDistance()
            mst, linkage_matrix, distance_df = distance_analyzer.analyze(df)

            ratios_analyzer = PortfolioRatios()
            ratios_df = ratios_analyzer.analyze(df)

            # Lưu kết quả vào session_state
            st.session_state[f"{filename}_autocorr_df"] = autocorr_df
            st.session_state[f"{filename}_autocov_df"] = autocov_df
            st.session_state[f"{filename}_distance_df"] = distance_df
            st.session_state[f"{filename}_mst"] = mst
            st.session_state[f"{filename}_linkage_matrix"] = linkage_matrix
            st.session_state[f"{filename}_ratios_df"] = ratios_df

    else:
        st.write("Vui lòng tải lên ít nhất một file CSV.")

            
with tab2:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            with st.expander('Correlation Analysis'):
                st.write('Plot 1')

                if "autocorr_df" in st.session_state:
                    autocorr_df = st.session_state["autocorr_df"]

                    fig, ax = plt.subplots(figsize=(6, 4))
                    for symbol in autocorr_df.columns:
                        ax.plot(autocorr_df.index, autocorr_df[symbol], label=symbol, marker='o', alpha=0.5, lw=1)
                    ax.set_title('Autocorrelation Plot')
                    ax.set_xlabel('Lag')
                    ax.set_ylabel('Autocorrelation')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig) 
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")

                st.write('Plot 4')

                if "autocorr_df" in st.session_state:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(autocorr_df.T, cmap="coolwarm", annot=False, ax=ax)
                    ax.set_title('Autocorrelation Heatmap', fontsize=12, weight='bold')
                    ax.set_xlabel('Lag', fontsize=10)
                    ax.set_ylabel('Symbols', fontsize=10)
                    st.pyplot(fig)  
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")

                st.write('Plot 7')

                if "autocov_df" in st.session_state:
                    autocov_df = st.session_state["autocov_df"]
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for symbol in autocov_df.columns:
                        ax.plot(autocov_df.index, autocov_df[symbol], label=symbol, marker='o', alpha=0.5, lw=1)
                    ax.set_title('Autocovariance Plot')
                    ax.set_xlabel('Lag')
                    ax.set_ylabel('Autocovariance')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)  
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")

                st.write('Plot 10')

                if "autocov_df" in st.session_state:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(autocov_df.T, cmap="coolwarm", annot=False, ax=ax)
                    ax.set_title('Autocovariance Heatmap', fontsize=12, weight='bold')
                    ax.set_xlabel('Lag', fontsize=10)
                    ax.set_ylabel('Symbols', fontsize=10)
                    st.pyplot(fig)  
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")

    with col2:
        with st.container(border=True):
            with st.expander('Distance'):
                st.write('Plot 2')
                if "distance_df" in st.session_state:
                    distance_df = st.session_state["distance_df"]
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(distance_df, annot=True, cmap="coolwarm", fmt=".1f", ax=ax)
                    ax.set_title("Distance Heatmap", fontsize=12, weight="bold")
                    ax.set_xlabel("Stocks", fontsize=10, weight="bold")
                    ax.set_ylabel("Stocks", fontsize=10, weight="bold")
                    st.pyplot(fig)  
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")

                st.write('Plot 5')
                if "mst" in st.session_state:
                    mst = st.session_state["mst"]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    pos = nx.spring_layout(mst, seed=42)
                    nx.draw(mst, pos, with_labels=True, node_size=2500, font_size=10, 
                            edge_color="blue", node_color="lightgreen", ax=ax)
                    ax.set_title("Minimum Spanning Tree", fontsize=12, weight="bold")
                    st.pyplot(fig)  
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")
                    
                st.write('Plot 8')
                if "linkage_matrix" in st.session_state:
                    linkage_matrix = st.session_state["linkage_matrix"]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    dendrogram(linkage_matrix, labels=distance_df.columns, leaf_rotation=90, leaf_font_size=10, ax=ax)
                    ax.set_title("Hierarchical Clustering Tree", fontsize=12, weight="bold")
                    st.pyplot(fig)
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")
        
    with col3:
        with st.container(border=True):
            with st.expander('Ratios'):
                st.write('Plot 3')
                if uploaded_files is not None:
                    portfolio_ratios = PortfolioRatios()
                    results_df, weights_df = portfolio_ratios.analyze(df)

                    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
                    metrics = ["Sharpe Ratio", "Treynor Ratio", "Jensen's Alpha", "Sortino Ratio"]

                    for i, metric in enumerate(metrics):
                        row, col = divmod(i, 2)
                        axes1[row, col].scatter(
                            results_df["Volatility"], results_df["Return"], c=results_df[metric],
                            cmap="viridis", marker="o", s=5, alpha=0.5
                        )
                        axes1[row, col].set_title(metric, fontsize=12, weight="bold")
                        axes1[row, col].set_xlabel("Volatility")
                        axes1[row, col].set_ylabel("Return")

                    plt.tight_layout()
                    st.pyplot(fig1)

                st.write('Plot 6')
                if uploaded_files is not None:
                    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 6))

                    for i, metric in enumerate(metrics):
                        best_idx = results_df[metric].idxmax()  # Tìm danh mục có chỉ số tốt nhất
                        best_weights = weights_df.iloc[best_idx]  # Lấy trọng số danh mục

                        row, col = divmod(i, 2)
                        axes2[row, col].bar(best_weights.index, best_weights, color='blue', alpha=0.8)
                        axes2[row, col].set_title(f"{metric} Portfolio Weights", fontsize=12, weight="bold")
                        axes2[row, col].set_xlabel("Asset")
                        axes2[row, col].set_ylabel("Weight")
                        axes2[row, col].tick_params(axis='x', rotation=90)

                    plt.tight_layout()
                    st.pyplot(fig2)


# with tab3:
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         with st.container(border = True):
#             with st.expander('Months'):
#                 st.write('Plot 1')
#                 st.write('Plot 4')
#     with col2:
#         with st.container(border = True):
#             with st.expander('Quarter'):
#                 st.write('Plot 2')
#                 st.write('Plot 5')
    
#     with col3:
#         with st.container(border = True):
#             with st.expander('Year'):
#                 st.write('Plot 3')
#                 st.write('Plot 6')        