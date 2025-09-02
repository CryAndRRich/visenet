import datetime
import os

TRAINING_DATA_FILE = "visenet/data/output/top_30_stocks_after_train.csv"

now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
os.makedirs(TRAINED_MODEL_DIR)