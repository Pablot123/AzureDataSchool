import pandas as pd
from azureml.core import Run, Workspace
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import argparse
import os

run = Run.get_context()

clean_data = run.input_datasets['clean_airlines_data']

parser = argparse.ArgumentParser("transform")
parser.add_argument("--output_transform", type=str, help="transformed data directory")

args = parser.parse_args()

print("Argument 3(output cleansed taxi data path): %s" % args.output_transform)


with mlflow.start_run():
    df = clean_data.to_pandas_dataframe()
    mlflow.log_metric('b', len(df.columns))
    numeric_columns = ['Length','Time']
    category_columns = ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                            ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))])

    preprocessor = ColumnTransformer(
                        transformers=[
                            ('numeric', numeric_transformer, numeric_columns),
                            ('cat', categorical_transformer, category_columns)
                        ],
                        remainder='passthrough'
                    )
    
    #split the dependent(y) and independent(df_x) features 
    df_X = df.drop('Class', axis=1, inplace=False)
    mlflow.log_metric('a', len(df_X.columns))
    #y_labels = data['Class']
    y = df['Class']
    preprocessed_data = preprocessor.fit_transform(df_X)
    encoded_category = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(category_columns)
    mlflow.log_metric('p', len(encoded_category))
    labels = np.concatenate([numeric_columns, encoded_category])
    
    preprocessed_df = pd.DataFrame(data=preprocessed_data, columns=labels)
    transsformed_df = pd.concat([preprocessed_df, y], axis=1)

    if not (args.output_transform is None):
        mlflow.log_metric('step_2', 222)
        os.makedirs(args.output_transform, exist_ok=True)
        print("%s created" % args.output_transform)
        path = args.output_transform + "/processed_t.csv"
        write_df = transsformed_df.to_csv(path)
        #transsformed_df.to_csv('transform.csv')
        mlflow.log_artifact(path)
