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

input_data = dbconn()

print("shape of data is " + str(input_data.shape) + "\n" )

print(input_data.head())

# Path to model storage location
MODEL_STORAGE_PATH = 'models/'

# Path to graph img storage location
GRAPH_STORAGE_PATH = 'graph/'

#%% PRE PROCESSING

data_train = input_data.copy()

# create the target column, whether ther was some burnt area or not
data_train['burnt'] = np.where(data_train['area']==0, 0, 1)

data_train = data_train.drop(['area'], axis=1)

# data_train = data_train.filter(['x', 'y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'burnt'], axis=1)

print(data_train.head())

# Separate columns by data type for analysis
cat_cols = [col for col in data_train.columns if data_train[col].dtype == object]
num_cols = [col for col in data_train.columns if data_train[col].dtype != object]

# Ensure that all columns have been accounted for
assert len(cat_cols) + len(num_cols) == data_train.shape[1]

# Look at cardinality of the categorical columns
cards = [len(data_train[col].unique()) for col in cat_cols]

# Convert missing values to string
data_train.replace(np.nan, 'NaN', inplace=True)
data_train.isna().sum()

# Create training and testing data
# x will be the measurements
# y will be whether there was burnt area or not
x, y = data_train.dropna().drop(['burnt'], axis=1), data_train.dropna()['burnt']

# 20 percent of records will be kept for testing 
# fixed random seed so process can be duplicated
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)
print("shape of training data is " + str(x_train.shape))
print("shape of test data is " + str(x_test.shape))

# Specify index of categorical features in input data
cat_features = [x.columns.get_loc(col) for col in cat_cols]

#%% VISUALIZATION

fig, ax = plt.subplots(figsize=(18, 6))
sns.barplot(x=cat_cols, y=cards)
ax.set_xlabel('Feature')
ax.set_ylabel('Number of Categories')
ax.set_title('Feature Cardinality')
# no month nor day of the week is free of fires

plt.show() 
plt.savefig(GRAPH_STORAGE_PATH +"feat_cardinality.png")

months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

fig, ax = plt.subplots(figsize=(18, 6))
sns.countplot(data=data_train, x='month', ax=ax, palette="pastel", hue='burnt', order=months)
ax.set_xlabel('Month')
ax.set_ylabel('Number of Categories')
ax.set_title('Ocurrences by month')

plt.show()
plt.savefig(GRAPH_STORAGE_PATH +"ocurrences_by_month.png")

# todo a plotly+dash or bokeh for an interactive dashboard

# Distribution of recorded events by burnt status
data_train_burnt = data_train[data_train['burnt'] == 1]
data_train_no_burnt = data_train[data_train['burnt'] == 0]

# Target distribution
fig, ax = plt.subplots(figsize=(10, 6))
print(data_train['burnt'].value_counts() / len(data_train))
sns.countplot(x=data_train['burnt'])
ax.set_ylabel('Burnt area?')
ax.set_ylabel('Number of ocurrences')
ax.set_title('Recorded events')
plt.savefig(GRAPH_STORAGE_PATH +"recorded_events.png")


#%% MODELING

# Model parameter dict
params = {'iterations': 5000,
          'loss_function': 'Logloss',
          'depth': 4,
          'early_stopping_rounds': 20,
          'custom_loss': ['AUC', 'Accuracy']}

# Instantiate model
model = CatBoostClassifier(**params)

# Fit model
model.fit(
    x_train,
    y_train,
    cat_features=cat_features,
    eval_set=(x_test, y_test),
    verbose=50,
    plot=True
)

model.save_model("saved_model.cbm")

# Make predictions on test data
preds = model.predict(x_test)

# Evaluate predictions
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

fig, ax = plt.subplots(figsize=(10, 6))

feature_importance_data = pd.DataFrame({'feature': model.feature_names_, 'importance': model.feature_importances_})
feature_importance_data.sort_values('importance', ascending=False, inplace=True)

sns.barplot(x='importance', y='feature', data=feature_importance_data)
ax.set_title('Importance of features')

plt.show()
plt.savefig(GRAPH_STORAGE_PATH +"importance_of_features.png")


#%% HYPER PARAMETER TUNING

def classification_objective(trial):
    params = {
        "loss_function": "Logloss",
        'custom_loss': ['Accuracy'],
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 1e0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1),
        "depth": trial.suggest_int("depth", 1, 10),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20)}

    if params["bootstrap_type"] == "Bayesian":

        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)

    elif params["bootstrap_type"] == "Bernoulli":

        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    # cross-validation for hyper-parameter tuning
    cv_data = cv(
        params=params,
        pool=Pool(x, label=y, cat_features=cat_features),
        fold_count=5,
        shuffle=True,
        partition_random_seed=0,
        plot=False,
        stratified=False,
        verbose=False)

    return cv_data['test-Accuracy-mean'].values[-1]

# tree-based hyper-parameter (Tree-structured Parzen Estimator)
classification_study = optuna.create_study(sampler=TPESampler(), direction="maximize")
classification_study.optimize(classification_objective, n_trials=20, timeout=600)
trial = classification_study.best_trial

print(f"Highest Accuracy: {trial.value}")
print("Optimal Parameters:")
for key, val in trial.params.items():
    print(f"{key}:{val}")
    
#%% BEST PARAM

# Create new parameter dictionary using optimal hyper-parameters
new_params = trial.params.copy()
new_params['loss_function'] = 'Logloss'
new_params['custom_loss'] = ['AUC','Accuracy']

cv_data = cv(
        params = new_params,
        pool = Pool(x, label=y, cat_features=cat_features),
        fold_count=5,
        shuffle=True,
        partition_random_seed=0,
        plot=False,
        stratified=False,
        verbose=False)

final_params = new_params.copy()

# The final number of iterations is iteration number that maximizes cross-validated accuracy
final_params['iterations'] = np.argmax(cv_data['test-Accuracy-mean'])
final_params['cat_features'] = cat_features

print(final_params)

final_model = CatBoostClassifier(**final_params)
final_model.fit(x,y,verbose=100)

# Export model
model_name = f'forest_fire_model_{str(datetime.today())[0:10]}'
final_model.save_model(MODEL_STORAGE_PATH + model_name)