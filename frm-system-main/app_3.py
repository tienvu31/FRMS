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
from arch import arch_model


st.title('FRM Analysis')

tab1, tab2, tab3 = st.tabs(['Table','Analysis','Prediction'])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.write('**Upload your dataset here**')
        with st.expander('Upload Dataset'):
            uploaded_file = st.file_uploader("Choose a file")
    with col2:
        if uploaded_file is not None:
            # Can be used wherever a "file-like" object is accepted:
            df = pd.read_csv(uploaded_file, thousands=',')  
            # df = df.replace(',', '', regex=True)  
            # df = df.apply(pd.to_numeric, errors='coerce')  
            # df['Date'] = pd.to_datetime(df['Date'])  
            # df.set_index('Date', inplace=True)
            st.write(df)  # Hiển thị dữ liệu sau khi xử lý
                
            analyzer = PortfolioAutoCorr()
            autocorr_df, autocov_df = analyzer.analyze(df)
            st.session_state["autocorr_df"] = autocorr_df
            st.session_state["autocov_df"] = autocov_df
            
            # Phân tích Distance
            distance_analyzer = PortfolioDistance()
            mst, linkage_matrix, distance_df = distance_analyzer.analyze(df)
            st.session_state["distance_df"] = distance_df
            st.session_state["mst"] = mst
            
            # garch_analyzer = PortfolioGarch()
            # returns = garch_analyzer.analyze(df)
            # st.session_state["returns"] = returns
            
            # portfolio_ratios_analyzer = PortfolioRatios()
            

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            with st.expander('Correlation Analysis'):
                st.write('Plot 1')

                if "autocorr_df" in st.session_state:
                    autocorr_df = st.session_state["autocorr_df"]

                    # --- Biểu đồ 1: Autocorrelation Plot ---
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for symbol in autocorr_df.columns:
                        ax.plot(autocorr_df.index, autocorr_df[symbol], label=symbol, marker='o', alpha=0.5, lw=1)
                    ax.set_title('Autocorrelation Plot')
                    ax.set_xlabel('Lag')
                    ax.set_ylabel('Autocorrelation')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)  # Hiển thị Plot 1
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")

                # --- Biểu đồ 2: Autocorrelation Heatmap ---
                st.write('Plot 4')

                if "autocorr_df" in st.session_state:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(autocorr_df.T, cmap="coolwarm", annot=False, ax=ax)
                    ax.set_title('Autocorrelation Heatmap', fontsize=12, weight='bold')
                    ax.set_xlabel('Lag', fontsize=10)
                    ax.set_ylabel('Symbols', fontsize=10)
                    st.pyplot(fig)  # Hiển thị Plot 2
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")

                # --- Biểu đồ 3: Autocovariance Plot ---
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
                    st.pyplot(fig)  # Hiển thị Plot 3
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")

                # --- Biểu đồ 4: Autocovariance Heatmap ---
                st.write('Plot 10')

                if "autocov_df" in st.session_state:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(autocov_df.T, cmap="coolwarm", annot=False, ax=ax)
                    ax.set_title('Autocovariance Heatmap', fontsize=12, weight='bold')
                    ax.set_xlabel('Lag', fontsize=10)
                    ax.set_ylabel('Symbols', fontsize=10)
                    st.pyplot(fig)  # Hiển thị Plot 4
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
                    st.pyplot(fig)  # Hiển thị Plot 2
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")

                # --- Thêm Plot 5 (MST - Minimum Spanning Tree) ---
                st.write('Plot 5')
                if "mst" in st.session_state:
                    mst = st.session_state["mst"]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    pos = nx.spring_layout(mst, seed=42)
                    nx.draw(mst, pos, with_labels=True, node_size=2500, font_size=10, 
                            edge_color="blue", node_color="lightgreen", ax=ax)
                    ax.set_title("Minimum Spanning Tree", fontsize=12, weight="bold")
                    st.pyplot(fig)  # Hiển thị Plot 5
                else:
                    st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")
            
    with col3:
        with st.container(border=True):
            with st.expander('GARCH Model Analysis'):
                st.write('Plot 3')
                # if "returns" in st.session_state:
                #     returns = st.session_state["returns"]
                    
                #     fig, axes = plt.subplots(nrows=(len(returns.columns) + 1) // 2, ncols=2, figsize=(12, 10))
                #     axes = axes.flatten()
                    
                #     for i, symbol in enumerate(returns.columns):
                #         model = arch_model(returns[symbol], vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
                #         res = model.fit(disp='off')
                        
                #         res.conditional_volatility.plot(ax=axes[i], color='red')
                #         axes[i].set_ylabel(r'$\sigma \times 10^3$', fontsize=14, weight='bold')
                #         axes[i].set_title(f'{symbol} - GARCH Volatility', fontsize=16, weight='bold')
                #         axes[i].grid(True)

                #     for j in range(i + 1, len(axes)):
                #         fig.delaxes(axes[j])

                #     plt.tight_layout()
                #     st.pyplot(fig)  # Hiển thị biểu đồ
                # else:
                #     st.write("Vui lòng tải lên file dữ liệu để hiển thị biểu đồ.")


                st.write('Plot 6')

                # Kiểm tra nếu dữ liệu đã được tải lên

                if df is not None:
                    portfolio_ratios_analyzer = PortfolioRatios()

                    # Đóng tất cả figure trước khi vẽ (tránh lỗi trùng lặp)
                    plt.close('all')

                    # Tạo hai figure riêng biệt từ portfolio_ratios.py
                    fig1, fig2 = portfolio_ratios_analyzer.visualize(df)

                    # Hiển thị biểu đồ đầu tiên (Efficient Frontier & Ratios) vào Plot 6
                    st.pyplot(fig1)  

                    # Hiển thị biểu đồ thứ hai (Portfolio Weights) vào Plot 9
                    st.pyplot(fig2)

                else:
                    st.warning("Please upload a dataset to perform Portfolio Analysis.")


                #     st.subheader("Plot 6: Portfolio Ratios Analysis")
                    
                #     # Gọi phương thức phân tích và trực quan hóa
                #     results_df, weights_df = portfolio_ratios_analyzer.analyze(df)
                #     fig = portfolio_ratios_analyzer.visualize(df)

                #     # Hiển thị hình ảnh
                #     st.pyplot(fig)
                # else:
                #     st.warning("Please upload a dataset to perform Portfolio Ratios Analysis.")
            
            # st.write('Plot 9')
            # if df is not None:
            #     portfolio_ratios_analyzer = PortfolioRatios()

            #     # Đóng tất cả các figure trước khi vẽ (Tránh lỗi PyplotGlobalUseWarning)
            #     plt.close('all')

            #     # Gọi lại visualize() để vẽ biểu đồ thứ 2
            #     portfolio_ratios_analyzer.visualize(df)

            #     # Lấy figure hiện tại (lần vẽ này sẽ là biểu đồ thứ 2)
            #     fig = plt.gcf()

            #     # Hiển thị figure trên Streamlit
            #     st.pyplot(fig)

            # else:
            #     st.warning("Please upload a dataset to perform Portfolio Weights Distribution.")





with tab3:
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border = True):
            with st.expander('Months'):
                st.write('Plot 1')
                st.write('Plot 4')
    with col2:
        with st.container(border = True):
            with st.expander('Quarter'):
                st.write('Plot 2')
                st.write('Plot 5')
    
    with col3:
        with st.container(border = True):
            with st.expander('Year'):
                st.write('Plot 3')
                st.write('Plot 6')        