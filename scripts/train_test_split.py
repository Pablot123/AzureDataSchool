from azureml.core import Run
import argparse
import mlflow
from sklearn.model_selection import train_test_split
from azureml.core import Dataset
import os
import pandas as pd


ws = Run.get_context().experiment.workspace

parser = argparse.ArgumentParser("split")
parser.add_argument('--input_data', type=str, help='data')
parser.add_argument("--output_train_data", type=str, help="train data directory")
parser.add_argument("--output_test_data", type=str, help="test data directory")
parser.add_argument("--output_val_data", type=str, help="validation data directory")
args = parser.parse_args()

transformed_data_path = args.input_data
transformed_data_file = os.path.join(transformed_data_path, 'transformed_data.csv')

def write_output(df, path, name_file):
    '''
    Escribe el dataset como csv en una ruta determinada
    '''
    os.makedirs(path, exist_ok=True)
    df.to_csv(path+name_file, index=False, header=True)
    mlflow.log_artifact(path+name_file)

with mlflow.start_run():

    transformed_df = pd.read_csv(transformed_data_file)
    train_data, test_data = train_test_split(transformed_df, test_size=0.20, random_state=73)
    train_data, val_data = train_test_split(train_data, train_size=0.80, random_state=37)
    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    val_data.reset_index(inplace=True, drop=True)

    #registro de los datasets
    datastore = ws.get_default_datastore()

    train_data_reg = Dataset.Tabular.register_pandas_dataframe(train_data,
                                                           target=datastore,
                                                           name='airlines_train_data')
    test_data_reg = Dataset.Tabular.register_pandas_dataframe(test_data,
                                                          target=datastore,
                                                          name='airlines_test_data')
    val_data_reg = Dataset.Tabular.register_pandas_dataframe(val_data,
                                                         target = datastore,
                                                         name='airlines_val_data')



    if not( args.output_train_data is None and args.output_test_data is None):

        write_output(train_data, args.output_train_data, '/train_data.csv')
        write_output(test_data, args.output_test_data, '/test_data.csv')
        write_output(val_data, args.output_val_data, '/val_data.csv')


