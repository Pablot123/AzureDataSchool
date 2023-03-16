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
    df = df[df['Airline'].isin(df['Airline'].value_counts().head(5).to_dict().keys())].reset_index(drop=True)
    
    if not (args.output_cleanse is None):
        os.makedirs(args.output_cleanse, exist_ok=True)
        print("%s created" % args.output_cleanse)
        path = os.path.join(args.output_cleanse, "clean_data.csv")
        write_df = df.to_csv(path,index=False, header=True)
        #df.to_csv('clean.csv')
        mlflow.log_artifact(path)
