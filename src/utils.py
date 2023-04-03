import pandas as pd
from azureml.core import Run, Workspace
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
import numpy as np
import joblib
import os

def dataset_transform(train_df=pd.DataFrame([]), val_df=pd.DataFrame([]), test_df=pd.DataFrame([])):
    
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

    if (not(train_df.empty) and not(val_df.empty) and test_df.empty):
        
        #transforming train data
        X_train = train_df.drop('Class', axis=1, inplace=False)
        y_train = train_df['Class']

        preprocessor.fit(X_train)
        preprocessed_data_train = preprocessor.transform(X_train)
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

        return transsformed_df_train, transsformed_df_val, preprocessor
    
    elif (train_df.empty and val_df.empty and not(test_df.empty)):
        X_test= test_df.drop('Class', axis=1, inplace=False)
        y_test = test_df['Class']

        preprocessed_data_test = preprocessor.fit_transform(X_test)
        encoded_category = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(category_columns)
        labels = np.concatenate([numeric_columns, encoded_category])
        
        preprocessed_df_test = pd.DataFrame(data=preprocessed_data_test, columns=labels)
        transsformed_df_test = pd.concat([preprocessed_df_test, y_test], axis=1)

        return transsformed_df_test
    
    else:
        return 'Worng combination'

def all_metrics(y_true, y_pred):
    '''
    Retorna las metricas de presicion, recall, accuracy y 
    la matriz de cofusion
    input: y_true: muestras reales de la clase
            y_pred: predicciones del modelo
    output: tupla con el valor de recall precision, accuracy y
            matriz de confusion respectivamente
    '''
    recall = recall_score(y_true,y_pred)
    precision = precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return recall, precision, acc, conf_matrix


def write_output(df, path, name_file):
    '''
    Escribe el dataset como csv en una ruta determinada
    '''
    os.makedirs(path, exist_ok=True)
    df.to_csv(path+name_file, index=False, header=True)
    mlflow.log_artifact(path+name_file)

def difference_btw_data(main_df:pd.DataFrame, new_df:pd.DataFrame):
    
    train_data = set(main_df.to_list())

    new_data = set(new_df.to_list())

    diff = list(new_data.difference(train_data))

    return diff
