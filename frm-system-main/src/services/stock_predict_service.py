class StockPredictService:
    def __init__(self, strategy):
        self._strategy = strategy

    def set_strategy(self, strategy):
        self._strategy = strategy

    def visualize(self, data):
        self._strategy.visualize(data)