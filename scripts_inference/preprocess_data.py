import pandas as pd
from azureml.core import Run, Workspace
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.utils import dataset_transform, dataset_transform_prueba
import numpy as np
import argparse
import os
import joblib

run = Run.get_context()

ws = run.experiment.workspace


parser = argparse.ArgumentParser("preprocess")
#parser.add_argument('--input_data', type=str, help='input data')
parser.add_argument('--output_data', type=str, help='output data')

args = parser.parse_args()

with mlflow.start_run():
    test_data = run.input_datasets['raw_data'].to_pandas_dataframe()
    #transsformed_df_test = dataset_transform(test_df=test_data)

    #test_data_X = test_data.drop('Class', axis=1, inplace=False)
    #test_data_y = test_data['Class']
    preprocess_name = 'scaler'
    preprocess = mlflow.sklearn.load_model(f'models:/{preprocess_name}/latest')

    numeric_columns = ['Length','Time']
    category_columns = ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']

    prueba = preprocess.transform(test_data)
    encoded_category = preprocess.named_transformers_['cat']['onehot'].get_feature_names_out(category_columns)
    labels = np.concatenate([numeric_columns, encoded_category])    
    prueba_df_all = pd.DataFrame(data=prueba, columns=labels)

    #prueba_df_all = pd.concat([prueba_df,test_data_y], axis=1)

    if not(args.output_data is None):
        os.makedirs(args.output_data, exist_ok=True)
        print("%s created" % args.output_data)
        path_test = args.output_data + "/processed_data.csv"
        write_df_val = prueba_df_all.to_csv(path_test, index=False, header=True)
        mlflow.log_artifact(path_test)
