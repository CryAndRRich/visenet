import datetime
import os
# data
#TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
TRAINING_DATA_FILE = "visenet/data/output/top_30_stocks_after_train.csv"

now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

TESTING_DATA_FILE = "test.csv"
