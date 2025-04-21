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

st.title('Financial Risk Management Analysis')

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
                st.session_state["datasets"][filename] = df
                st.write(f"**Dataset: {filename}**")
                st.write(df)

with tab2:
    col1, col2, col3 = st.columns(3)
    # ---- Column 1: EDA ----
    with col1:
        with st.container(border=True):
            with st.expander("Exploratory Data Analysis"):
                st.write("Scatter Plot, Regression Line & Correlation Heatmap")

                if "datasets" in st.session_state and st.session_state["datasets"]:
                    eda_analyzer = PortfolioEDA()

                    for filename, df in st.session_state["datasets"].items():
                        st.markdown(f"**Dataset: {filename}**")
                        eda_analyzer.visualize(df)
    # ---- Column 2: Distance ----
    with col2:
        with st.container(border=True):
            with st.expander('Distance Analysis'):
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    strategy = PortfolioDistance()

                    for filename in st.session_state["datasets"].keys():
                        df = st.session_state["datasets"][filename]
                        st.markdown(f"**Distance-based Clustering - {filename}**")
                        strategy.visualize(df)
    # ---- Column 3: Stationary ----  
    with col3:
        with st.container(border=True):
            with st.expander('Stationarity Analysis'):
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    stationary = PortfolioStationary()

                    for filename in st.session_state["datasets"].keys():
                        df = st.session_state["datasets"][filename]
                        st.markdown(f"**St - {filename}**")
                        stationary.visualize(df)  
                         
    col1, col2, col3 = st.columns(3)
    # ---- Column 1: Correlation ----
    with col1:
        with st.container(border=True):
            with st.expander('Correlation Analysis'):
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    strategy = PortfolioAutoCorr()

                    for filename in st.session_state["datasets"].keys():
                        df = st.session_state["datasets"][filename]

                        st.markdown(f"**Autocorrelation & Autocovariance - {filename}**")
                        strategy.visualize(df)
    # ---- Column 2: GARCH ----
    with col2:
        with st.container(border=True):
            with st.expander('Volatility Analysis'):
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    garch_analyzer = PortfolioGarch()

                    for filename, df in st.session_state["datasets"].items():
                        st.markdown(f"**Dataset: {filename}**")
                        garch_analyzer.visualize(df)
    # ---- Column 3： Spectral Density ----
    with col3:
        with st.container(border = True):
            with st.expander('Spectral Density Analysis'):
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    spectral_analyzer = PortfolioSpectralDensity()

                    for filename in st.session_state["datasets"].keys():
                        df = st.session_state["datasets"][filename]
                        st.markdown(f"**Density - {filename}**")
                        spectral_analyzer.visualize(df)

    col1, col2, col3 = st.columns(3)
    # ---- Column 1: Portfolio ----                     
    with col1:
        with st.container(border=True):
            with st.expander('Portfolio Analysis'):
                if "datasets" in st.session_state and st.session_state["datasets"]:
                    ratios = PortfolioRatios()

                    for filename in st.session_state["datasets"].keys():
                        df = st.session_state["datasets"][filename]
                        st.markdown(f"**St - {filename}**")
                        ratios.visualize(df) 

with tab3:
    rnn_strategy = StockRNNStratgy()

    if "datasets" in st.session_state and st.session_state["datasets"]:
        for filename, df in st.session_state["datasets"].items():
            with st.expander(f"Prediction for: {filename}"):
                try:
                    rnn_strategy.visualize(df)
                except Exception as e:
                    st.error(f"Error in processing {filename}: {e}")
    else:
        st.warning("Please upload your dataset")
