import pandas as pd
from azureml.core import Run, Workspace
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.utils import dataset_transform
import numpy as np
import argparse
import os

run = Run.get_context()

parser = argparse.ArgumentParser("transform")
parser.add_argument('--input_data_train', type=str, help='train data')
parser.add_argument('--input_data_val', type=str, help='val data')
parser.add_argument('--input_data_test', type=str, help='val data')


parser.add_argument("--output_transform_train", type=str, help="transformed data train")
parser.add_argument("--output_transform_val", type=str, help="transformed data test")
parser.add_argument("--output_transform_test", type=str, help="transformed data train")

args = parser.parse_args()


with mlflow.start_run():
    
    if not (args.output_transform_train == 'None' and args.output_transform_val == 'None'):
        clean_data_train_path = args.input_data_train
        clean_data_train_file = os.path.join(clean_data_train_path, 'train_data.csv')

        clean_data_val_path = args.input_data_val
        clean_data_val_file = os.path.join(clean_data_val_path, 'val_data.csv')

        train_df = pd.read_csv(clean_data_train_file)
        val_df = pd.read_csv(clean_data_val_file)
    
        transsformed_df_train, transsformed_df_val = dataset_transform(train_df = train_df, val_df=val_df)

        #saving train data transformed
        os.makedirs(args.output_transform_train, exist_ok=True)
        print("%s created" % args.output_transform_train)
        path_train = args.output_transform_train + "/transformed_data_train.csv"
        write_df_train = transsformed_df_train.to_csv(path_train, index=False, header=True)
        mlflow.log_artifact(path_train)

        #saving val data transformed
        os.makedirs(args.output_transform_val, exist_ok=True)
        print("%s created" % args.output_transform_val)
        path_val = args.output_transform_val + "/transformed_data_val.csv"
        write_df_val = transsformed_df_val.to_csv(path_val, index=False, header=True)
        mlflow.log_artifact(path_val)

    else:
        test_data = run.input_datasets['raw_data_test'].to_pandas_dataframe()
        transsformed_df_test = dataset_transform(test_df=test_data)

        os.makedirs(args.output_transform_test, exist_ok=True)
        print("%s created" % args.output_transform_test)
        path_test = args.output_transform_test + "/transformed_data_test.csv"
        write_df_val = transsformed_df_test.to_csv(path_test, index=False, header=True)
        mlflow.log_artifact(path_test)



