import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.common.consts import CommonConsts
from src.utils.logger import LOGGER


class ModelTrainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
    
    def train_model(self, train_loader: torch.utils.data.DataLoader) -> None:
        for epoch in range(CommonConsts.EPOCHS):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            LOGGER.info(f'Epoch {epoch+1}/{CommonConsts.EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}')
    
    def predict(self, X_test: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).numpy()
        return predictions

    def predict_future_price(self, last_sequence: np.ndarray) -> np.ndarray:
        self.model.eval()
        future_predictions = []
        # temp_sequence = np.copy(last_sequence)
        
        for _ in range(CommonConsts.FORECAST_DAYS):
            with torch.no_grad():
                future_pred = self.model(
                    torch.tensor(last_sequence, dtype=torch.float32)
                ).numpy()
                future_predictions.append(future_pred[0, 0])
                last_sequence = np.append(
                    last_sequence[:, 1:, :],
                    future_pred.reshape(1, 1, 1),
                    axis=1
                )
        
        return np.array(future_predictions).reshape(-1, 1)