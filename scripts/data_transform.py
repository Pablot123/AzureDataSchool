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

#clean_data = run.input_datasets['cleaned_data']

parser = argparse.ArgumentParser("transform")
parser.add_argument('--input_data_train', type=str, help='train data')
parser.add_argument('--input_data_val', type=str, help='val data')

parser.add_argument("--output_transform_train", type=str, help="transformed data train")
parser.add_argument("--output_transform_val", type=str, help="transformed data test")

args = parser.parse_args()

clean_data_train_path = args.input_data_train
clean_data_train_file = os.path.join(clean_data_train_path, 'train_data.csv')

clean_data_val_path = args.input_data_val
clean_data_val_file = os.path.join(clean_data_val_path, 'val_data.csv')


with mlflow.start_run():
    train_df = pd.read_csv(clean_data_train_file)
    val_df = pd.read_csv(clean_data_val_file)
    
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
    
    #transforming train data
    X_train = train_df.drop('Class', axis=1, inplace=False)
    y_train = train_df['Class']

    preprocessed_data_train = preprocessor.fit_transform(X_train)
    encoded_category = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(category_columns)
    labels = np.concatenate([numeric_columns, encoded_category])
    
    preprocessed_df_train = pd.DataFrame(data=preprocessed_data_train, columns=labels)
    transsformed_df_train = pd.concat([preprocessed_df_train, y_train], axis=1)

    #transforming validation data
    X_val = val_df.drop('Class', axis=1, inplace=False)
    y_val = val_df['Class']

    preprocessed_data_val = preprocessor.transform(X_val)
    encoded_category = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(category_columns)
    labels = np.concatenate([numeric_columns, encoded_category])
    
    preprocessed_df_val = pd.DataFrame(data=preprocessed_data_val, columns=labels)
    transsformed_df_val = pd.concat([preprocessed_df_val, y_val], axis=1)



    if not (args.output_transform_train is None and args.output_transform_val is None):
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
