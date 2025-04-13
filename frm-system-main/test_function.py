import pandas as pd

from src.common.consts import CommonConsts
from src.data.processors import Processors
from src.services.statistical_analysis_service import StatisticalAnalysisService
from src.services.stock_predict_service import StockPredictService
from src.services.strategies import (
    PortfolioEDA,
    PortfolioAutoCorr,
    PortfolioSpectralDensity,
    PortfolioDistance,
    PortfolioGarch,
    PortfolioRatios,
    PortfolioStationary
)
from src.services.strategies import StockRNNStratgy
from src.common.consts import CommonConsts


'''
Cách test các strategy:
1. Chọn strategy cần test
    VD: strategy = PortfolioEDA()
2. Tạo service với strategy đã chọn 
    VD: - với các phương pháp phân tích như EDA hay AutoCorr
          thì gán: service = StatisticalAnalysisService(strategy=strategy)
        - với phương pháp dự đoán giá cổ phiếu
          thì gán: service = StockPredictService(strategy=strategy)
3. Gọi hàm visualize để trực quan hóa và lưu ảnh vào thư mục img
    VD: service.visualize(df)


Hướng phát triển:
- Thêm các strategy, mô hình dự đoán giá mới vào thư mục src/services/strategies
- Lấy data mới nhất theo ngày hoặc realtime từ API hoặc database
- Phát triển API để build app
'''

data_path = "frm-system-main/src/data/fiinx/data_it_2021.csv" # Data tĩnh
data_2021 = pd.read_csv(data_path)
# data_2020 = CommonConsts.ticker_model
df = Processors.transform(data_2021)

# Chọn strategy cần test, comment các strategy không cần test 
# analysis_strategy = PortfolioEDA()
# analysis_strategy = PortfolioAutoCorr()
# analysis_strategy = PortfolioSpectralDensity()
# analysis_strategy = PortfolioDistance()
analysis_strategy = PortfolioGarch()
# analysis_strategy = PortfolioRatios()
# analysis_strategy = PortfolioStationary()

predict_stock_strategy = StockRNNStratgy()
 
# Test strategy phân tích portfolio
service = StatisticalAnalysisService(strategy=analysis_strategy)
service.visualize(df)

# Test strategy dự đoán giá cổ phiếu
service = StockPredictService(strategy=predict_stock_strategy)
service.visualize(df)
