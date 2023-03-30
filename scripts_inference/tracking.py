import mlflow
from azureml.core import Workspace, Datastore, Run, Dataset
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from src.utils import dataset_transform
import argparse
import os


run = Run.get_context()

ws = run.experiment.workspace

with mlflow.start_run():
    #data from training dataset
    train_data = Dataset.get_by_name(ws, name='airlines_train_data').to_pandas_dataframe()
    

    #data from new data
    test_data = run.input_datasets['raw_data'].to_pandas_dataframe()
    print('hola')