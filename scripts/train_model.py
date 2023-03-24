import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from azureml.core import Run
import pandas as pd
import numpy as np
import argparse
import os
import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('Img_To_Local_Python.log', 'w', 'utf-8')
root_logger.addHandler(handler)


parser = argparse.ArgumentParser("train")
parser.add_argument('--input_data_train', type=str, help='training data')
parser.add_argument('--input_data_val', type=str, help='validation data')

parser.add_argument('--output_model', type=str, help='trained model' )


args = parser.parse_args()

train_data_path = args.input_data_train
train_data_file = os.path.join(train_data_path, 'transformed_data_train.csv')

val_data_path = args.input_data_val
val_data_file = os.path.join(val_data_path, 'transformed_data_val.csv')

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


with mlflow.start_run() as run:

    train_df = pd.read_csv(train_data_file)
    X_train = train_df.drop('Class', axis=1, inplace=False)
    y_train = train_df['Class']

    val_df = pd.read_csv(val_data_file)
    X_val = val_df.drop('Class', axis=1, inplace=False)
    y_val = val_df['Class']


    param_dist ={
        'penalty':['l1', 'l2', 'elasticnet'],
        'C': [c/10 for c in range(1,20,5)],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']

    }
    clf = RandomizedSearchCV(LogisticRegression(), param_dist, n_iter=5, random_state=0, refit='balanced_accuracy')
   
    clf.fit(X_train,y_train)


    
    mlflow.log_params(clf.best_params_)

    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)


    recall_train, precision_train, acc_train, conf_matrix_train = all_metrics(y_train, y_pred_train)
    recall_val, precision_val, acc_val, conf_matrix_val = all_metrics(y_val, y_pred_val)

    
    mlflow.log_metrics({
        'recall train':recall_train,
        'precision train': precision_train,
        'accuracy train': acc_train,
        'recall validation': recall_val,
        'precision validation': precision_val,
        'accuracy validation': acc_val

    })
    #model_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mlflow.sklearn.log_model(sk_model = clf,
                             artifact_path = "sklearn-model")

    
    if not( args.output_model is None ):
        model_path = os.path.join(args.output_model, 'sklearn-model')
        mlflow.sklearn.save_model(sk_model=clf,
                                  path=model_path)


    
