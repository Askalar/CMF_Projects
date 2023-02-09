import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import torch
from datetime import datetime, timedelta
from yahoo_fin.stock_info import get_data
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco
from fedot.api.main import Fedot
pd.options.mode.chained_assignment = None 



def company_cross_validation(data: pd.DataFrame, name_of_column: str, model, metrics, train_lenght: int=0, step: int=1, plot: bool=False) -> float:
    """Cross validation for one company data"""
    values_for_metric = {"test":[], "model":[]}
    len_of_data = len(data[name_of_column])
    data['Time'] = np.arange(len(data.date))
    data = data.reset_index(drop=True)
    if not train_lenght:
        train_lenght = int((len_of_data * 0.6)//1)
    for n in range(train_lenght + 1, len_of_data, step):
        train_data = data.iloc[: n]
        values_for_metric["model"].append(model(train_data, name_of_column))
        values_for_metric["test"].append(data[name_of_column][n])
    if plot:
        plt.figure(figsize=(20,10))
        plt.plot(values_for_metric["model"], label='Predicted')
        plt.plot(values_for_metric["test"], label='Actual')
        plt.title("Predicted and actual values for " + model.__name__ + ". Name of company is " + data.company_name[0] + ".")
        plt.legend()
        plt.grid()
        plt.show()
    return metrics(values_for_metric["test"], values_for_metric["model"])


def companies_validation(data: pd.DataFrame, name_of_column: str, 
                        model, metrics, train_lenght: int=0, 
                        step: int=1, plot: bool=False) -> pd.DataFrame:
    """Cross validation for a few companies data"""
    validation = {}
    for company_name in data.company_name.unique():
        DF_of_one_company = data.loc[data.company_name == company_name] 
        validation[company_name] = company_cross_validation(DF_of_one_company, name_of_column, model, metrics, plot = plot)
    return pd.DataFrame.from_dict(validation, orient='index', columns=[name_of_column])


def linear_model(data: pd.DataFrame, name_of_column: str) -> float:
    """Least squares Linear Regression."""
    model = LinearRegression().fit(np.array(data.Time[:]).reshape((-1,1)), data[name_of_column][:])
    next_value = model.coef_[0] * (data.Time[:].max() + 1) + model.intercept_ 
    return next_value


def multilinear_model(data: pd.DataFrame, name_of_column: str) -> float:
    """Least squares Linear Regression with multiply variables."""
    data["future_value"] = data[name_of_column].shift(-1)
    row_to_predict = data.tail(1)
    data = data.dropna()
    model = LinearRegression().fit(data[["rev","op_in","usd","Time"]], data["future_value"][:])
    next_value = model.predict(row_to_predict[["rev","op_in","usd","Time"]])[0]
    return next_value


def naive_model(data: pd.DataFrame, name_of_column: str) -> float:
    return data[name_of_column].iloc[-1]


def random_forest_model(data: pd.DataFrame, name_of_column: str):
    """."""
    data["future_value"] = data[name_of_column].shift(-1)
    row_to_predict = data[data.columns.difference(['future_value', 'company_name', 'date'])].tail(1)
    row_to_predict_scaled = StandardScaler().fit_transform(row_to_predict)
    data = data.dropna()
    data_X = data[data.columns.difference(['future_value', 'company_name', 'date'])]
    data_X_scaled = StandardScaler().fit_transform(data_X)
    data_Y = np.array(data["future_value"])
    forest_model = RandomForestRegressor()
    forest_model.fit(data_X_scaled, data_Y)   
    next_value = forest_model.predict(row_to_predict_scaled)
    return next_value



def AML_model(data: pd.DataFrame, name_of_column: str, recalculate: bool = False) -> float:
    """Least squares Linear Regression with multiply variables."""
    global calculated, automl_parameters 
    if not calculated or recalculate:
        N_THREADS = 4
        N_FOLDS = 5
        RANDOM_STATE = 42
        TEST_SIZE = 0.2
        TIMEOUT = 300
        TARGET_NAME = 'TARGET'
        calculated = True
        task = Task('reg')
        automl_parameters = TabularAutoML(task = task, timeout = TIMEOUT, cpu_limit = N_THREADS,
                                reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE})
        roles = {'target': name_of_column}
        AML_fit_predict = automl_parameters.fit_predict(data[["rev","op_in","usd"]], roles = roles, verbose = 1)
    next_value = automl_parameters.predict(data[["rev","op_in","usd"]])[0].data[0]
    return next_value

