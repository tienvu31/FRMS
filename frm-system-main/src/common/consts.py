import os

class CommonConsts:

    ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 2 * "../"))
    IMG_FOLDER = os.path.join(ROOT_FOLDER, "src\imgs")
    ticker_model = [
        "BID",
        "VCB",
        "CTG",
        "MBB",
        "ACB",
        "STB",
    ]

    SEQUENCE_LENGTH: int = 63
    HIDDEN_SIZE: int = 32
    NUM_LAYERS: int = 4
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 100
    FORECAST_DAYS: int = 63  # 3 months - weekdays
