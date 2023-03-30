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

def dataset_transform_prueba(train_df=pd.DataFrame([]), val_df=pd.DataFrame([]), test_df=pd.DataFrame([])):

    numeric_columns = ['Length','Time']
    category_columns = ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']
    
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    sc = StandardScaler()

    if (not(train_df.empty) and not(val_df.empty) and test_df.empty):
        
        #transforming train data
        X_train = train_df.drop('Class', axis=1, inplace=False)
        y_train = train_df['Class']

        X_train_oh = oh_encoder.fit_transform(X_train[category_columns])
        X_train_sc = sc.fit_transform(X_train[numeric_columns])
        one_hot_columns_name = oh_encoder.get_feature_names_out()

        train_oh_df = pd.DataFrame(data = X_train_oh, columns = one_hot_columns_name)
        train_sc_df = pd.DataFrame(data = X_train_sc, columns=numeric_columns)

        train_df = pd.concat([train_sc_df, train_oh_df, y_train], axis=1)

        #transformig val data
        X_val = val_df.drop('Class', axis=1, inplace=False)
        y_val = val_df['Class']

        X_val_oh = oh_encoder.transform(X_val[category_columns])
        X_val_sc = sc.transform(X_val[numeric_columns])
        one_hot_columns_name = oh_encoder.get_feature_names_out()

        val_oh_df = pd.DataFrame(data = X_val_oh, columns = one_hot_columns_name)
        val_sc_df = pd.DataFrame(data = X_val_sc, columns=numeric_columns)

        val_df = pd.concat([val_sc_df, val_oh_df, y_val], axis=1)

        return train_df, val_df, oh_encoder, sc
    
    elif (train_df.empty and val_df.empty and not(test_df.empty)):
        test_df.reset_index(inplace=True)
        
        X_test= test_df.drop('Class', axis=1, inplace=False)
        y_test = test_df['Class']
      
        sc_t = joblib.load('standarScaler.joblib')
        oh_t = joblib.load('oh_encoder.joblib')

        sc_data_test = sc_t.transform(X_test[numeric_columns])
        oh_data_test = oh_t.transform(X_test[category_columns])
        one_hot_columns_name_test = oh_t.get_feature_names_out()

        test_oh_df = pd.DataFrame(data = oh_data_test, columns = one_hot_columns_name_test)
        test_sc_df = pd.DataFrame(data = sc_data_test, columns=numeric_columns)

        test_df_p = pd.concat([test_sc_df, test_oh_df, y_test], axis=1)

        return test_df_p
    
    else:
        return 'Worng combination'