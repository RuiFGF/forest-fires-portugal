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

#%% PRE PROCESSING

data_train = input_data.copy()

# change numerical id to a more intuitive one
# data_train['uid'] = data_train['x'].astype(str) + data_train['y'].astype(str) + data_train['month'].astype(str)

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
#x, y = data_train
x, y = data_train.dropna().drop(['burnt'], axis=1), data_train.dropna()['burnt']

# 20 percent of records will be kept for testing with fixed random seed
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

# Mosaic plot with gender distribution
# labelizer = lambda k: \
# {('M', '1'): 'Male w/ HD', ('F', '1'): 'Female w/ HD', ('M', '0'): 'Male w/o HD', ('F', '0'): 'Female w/o HD'}[k]
# mosaic(data_train, ['Sex', 'HeartDisease'], labelizer=labelizer,
       # title='Heart Disease by Gender, proportions in study')

# Distribution of bp by disease status
fig, ax = plt.subplots(figsize=(10, 6))
data_train_disease = data_train[data_train['burnt'] == 1]
data_train_no_disease = data_train[data_train['burnt'] == 0]

# sns.histplot(x=data_train_no_disease['RestingBP'], label='No disease', ax=ax, kde=True, color="green")
# sns.histplot(x=data_train_disease['RestingBP'], label='Disease', ax=ax, kde=True, color="red")
# plt.legend(['No disease', 'Disease'])
# ax.set_title('Resting BP Distribution')

plt.show()

# Target distribution
fig, ax = plt.subplots(figsize=(10, 6))
print(data_train['burnt'].value_counts() / len(data_train))
sns.countplot(x=data_train['burnt'])
ax.set_ylabel('Área queimada?')
ax.set_ylabel('Number de ocurrências')
ax.set_title('Ocurências registadas')


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

plt.show()


#%% HYPER PARAMETER

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