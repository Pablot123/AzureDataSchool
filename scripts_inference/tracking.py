import mlflow
from azureml.core import Workspace, Datastore, Run, Dataset
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from src.utils import difference_btw_data
import json
import argparse
import os


run = Run.get_context()

ws = run.experiment.workspace

with mlflow.start_run():
    category_columns = ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']

    #data from training dataset
    train_data = Dataset.get_by_name(ws, name='airlines_train_data').to_pandas_dataframe()

    #data from new data
    new_data = run.input_datasets['raw_data'].to_pandas_dataframe()
    
    airline_diff = difference_btw_data(train_data['Airline'], new_data['Airline'])
    airportfrom_diff = difference_btw_data(train_data['AirportFrom'], new_data['AirportFrom'])
    airportto_diff = difference_btw_data(train_data['AirportTo'], new_data['AirportTo'])
    dayweek_diff = difference_btw_data(train_data['DayOfWeek'], new_data['DayOfWeek'])

    metrics_report= {
        'number of new data Airline column':len(airline_diff),
        'number of new data AirportFrom column':len(airportfrom_diff),
        'number of new data AirportTo column':len(airportto_diff),
        'number of new data DayOfWeek column':len(dayweek_diff)      
    }

    mlflow.log_metrics(metrics_report)
    
    new_data_report = {
        'Airline':airline_diff,
        'AirportFrom':airportfrom_diff,
        'AirportTo':airportto_diff,
        'DayOfWeek':dayweek_diff
    }
    
    mlflow.log_dict(new_data_report, 'new_data_report.json')
    