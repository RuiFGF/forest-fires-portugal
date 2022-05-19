import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

from connection import dbconn

from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import optuna
from optuna.samplers import TPESampler
from datetime import datetime

# display all of the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

# %matplotlib inline

data_train = dbconn()

print("shape of data is " + str(data_train.shape) + "\n" )

print(data_train.head())

# Path to model storage location
MODEL_STORAGE_PATH = 'models/'