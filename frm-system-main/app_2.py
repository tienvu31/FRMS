import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from src.services.strategies.stock_predictions.stock_rnn_strategy import StockRNNStratgy
from src.data.processors import Processors
from src.utils.logger import LOGGER
from src.services.stock_predict_service import StockPredictService
from src.services.strategies.stock_predictions.models import StockRNN
from src.services.strategies.stock_predictions.trainers import ModelTrainer
from src.common.consts import CommonConsts

st.set_page_config(page_title="KLTN-FRM Analysis", layout="wide")

st.title('FRM Analysis')

tab1, tab2, tab3 = st.tabs(['Table','Analysis','Prediction'])

# Lưu datasets vào session_state
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}

    # uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    # if uploaded_file is not None:
    #     df = pd.read_csv(uploaded_file)
    #     if isinstance(df, pd.DataFrame):  # Kiểm tra có phải DataFrame không
    #         st.session_state["datasets"] = df  # Lưu đúng vào session_state
    #         st.write("Dataset loaded successfully:", df.head())
    #         st.write(df.head(10).to_dict())  # Kiểm tra toàn bộ dữ liệu

    #         if "VCB" in df.columns:
    #             st.write("VCB column exists!")
    #             st.write(df["VCB"].head())  # Hiển thị 5 dòng đầu
    #             st.write("VCB column dtype:", df["VCB"].dtype)
    #             df["VCB"] = pd.to_numeric(df["VCB"], errors="coerce")
    #             st.write("VCB after conversion:", df["VCB"].head(10))

    #         else:
    #             st.error("Stock symbol 'VCB' not found in dataset!")

    #     else:
    #         st.error("Uploaded file is not a valid DataFrame.")
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

# with tab3:
#     st.subheader("Stock Price Prediction using RNN")
    
#     # Chọn dataset và mã cổ phiếu để dự đoán
#     if st.session_state["datasets"]:
#         selected_dataset = st.selectbox("Select Dataset", list(st.session_state["datasets"].keys()))
#         df = st.session_state["datasets"][selected_dataset]
#         available_stocks = df.columns[1:]  # Bỏ cột 'Date'
#         selected_stocks = st.multiselect("Select Stocks to Predict", available_stocks)
        
#         if selected_stocks:
#             model_strategy = StockRNNStratgy()
            
#             for stock in selected_stocks:
#                 with st.expander(f"Prediction for {stock}"):
#                     fig, ax = plt.subplots(figsize=(10, 4))
                    
#                     try:
#                         predicted_prices = model_strategy.predict(df, stock)
#                         ax.plot(predicted_prices.index, predicted_prices, label="Predicted", color='red')
#                         ax.plot(df["Date"], df[stock], label="Actual", color='blue', alpha=0.6)
#                         ax.set_title(f"Predicted vs Actual Prices - {stock}")
#                         ax.set_xlabel("Date")
#                         ax.set_ylabel("Price")
#                         ax.legend()
#                         st.pyplot(fig)
#                     except Exception as e:
#                         st.error(f"Error in prediction for {stock}: {e}")
#         else:
#             st.warning("Please select at least one stock to predict.")
#     else:
#         st.warning("No dataset uploaded. Please upload a dataset in the 'Table' tab.")

with tab3:
    st.write('**Prediction**')
    
    for filename, df in st.session_state.get("datasets", {}).items():
        st.write(f"### Predicting for {filename}")
        
        stock_rnn_strategy = StockRNNStratgy()
        predictions = stock_rnn_strategy.visualize(df)

        # Kiểm tra xem predictions có hợp lệ không
        if not predictions:
            st.warning(f"No predictions generated for {filename}.")
            continue  # Bỏ qua nếu không có dự đoán

        for symbol, pred_df in predictions.items():
            with st.expander(f"Stock: {symbol}"):
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))

                # Test set predictions
                axes[0].plot(pred_df['Actual'], label='Actual Prices', color='blue')
                axes[0].plot(pred_df['Predicted'], label='Predicted Prices', marker='o', alpha=0.5, color='red')
                axes[0].legend()
                axes[0].set_title(f'{symbol} Price Prediction (Test Set)', weight='bold')
                axes[0].set_xlabel('Time [days]', fontsize=12, weight='bold')
                axes[0].set_ylabel('Value [USD]', fontsize=12, weight='bold')
                axes[0].grid(True)

                # Future predictions
                axes[1].plot(pred_df['Future_Predicted'], label='Predicted Future Prices', color='red', alpha=0.8)
                axes[1].legend()
                axes[1].set_title(f'{symbol} 3-Month Price Forecast', weight='bold')
                axes[1].set_xlabel('Time [days]', fontsize=12, weight='bold')
                axes[1].set_ylabel('Value [USD]', fontsize=12, weight='bold')
                axes[1].grid(True)

                st.pyplot(fig)
