import pandas as pd
from azureml.core import Run, Workspace
import mlflow

import argparse
import os
run = Run.get_context()

raw_data = run.input_datasets['raw_data']

parser = argparse.ArgumentParser("cleanse")
parser.add_argument("--output_cleanse", type=str, help="cleaned data directory")

args = parser.parse_args()

print("Argument 3(output cleansed taxi data path): %s" % args.output_cleanse)


with mlflow.start_run():

    df = raw_data.to_pandas_dataframe().drop('Flight', axis=1, inplace=False)
    
    if not (args.output_cleanse is None):
        mlflow.log_metric('step1', 111)
        os.makedirs(args.output_cleanse, exist_ok=True)
        print("%s created" % args.output_cleanse)
        path = args.output_cleanse + "/processed.csv"
        write_df = df.to_csv(path,index=False)
        #df.to_csv('clean.csv')
        mlflow.log_artifact(path)
