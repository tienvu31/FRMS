import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from src.common.consts import CommonConsts
class Processors:
    def transform(df):
        df = df.dropna(axis=0)
        df = df.drop_duplicates()
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].str.replace(',', '').astype(float)

        return df
    
    def prepare_data(symbol_df):
        data = symbol_df.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data)

        sequence_length = CommonConsts.SEQUENCE_LENGTH # Use 63 days of data to predict the next value
        X, y = [], []

        for i in range(len(data_normalized) - sequence_length):
            X.append(data_normalized[i:i+sequence_length])
            y.append(data_normalized[i+sequence_length])

        X = np.array(X)
        y = np.array(y) 

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        return {
            'train_loader': DataLoader(
                list(zip(X_train, y_train)),
                batch_size=CommonConsts.BATCH_SIZE,
                shuffle=True
            ),
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler
        }