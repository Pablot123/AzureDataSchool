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
parser.add_argument('--input_data_val', type=str, help='validation_data')

parser.add_argument('--output_model', type=str, help='trained model' )


args = parser.parse_args()

train_data_path = args.input_data_train
train_data_file = os.path.join(train_data_path, 'train_data.csv')

val_data_path = args.input_data_val
val_data_file = os.path.join(val_data_path, 'val_data.csv')


with mlflow.start_run() as run:
    df = pd.read_csv(train_data_file)

    x = df.drop('Class', axis=1, inplace=False)
    y = df['Class']


    param_dist ={
        'penalty':['l1', 'l2', 'elasticnet'],
        'C': [c/10 for c in range(1,20,5)],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']

    }
    clf = RandomizedSearchCV(LogisticRegression(), param_dist, n_iter=5, random_state=0, refit='balanced_accuracy')
   
    clf.fit(x,y)


    
    mlflow.log_params(clf.best_params_)

    y_pred = clf.predict(x)

    recall = recall_score(y,y_pred)
    precision = precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
  

    logging.info(f'matriz de confusion{conf_matrix}')

    
    mlflow.log_metrics({
        'recall':recall,
        'precision': precision,
        'accuracy': acc,

    })
    #model_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mlflow.sklearn.log_model(sk_model = clf,
                             artifact_path = "sklearn-model")

    
    if not( args.output_model is None ):
        model_path = os.path.join(args.output_model, 'sklearn-model')
        mlflow.sklearn.save_model(sk_model=clf,
                                  path=model_path)


    
