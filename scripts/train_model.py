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
#run = Run.get_context()
#train_data = run.input_datasets['output_split_train']
#train_df = train_data.to_pandas_dataframe()

parser = argparse.ArgumentParser("train")
parser.add_argument('--input_data', type=str, help='data to transform')
parser.add_argument("--output_transform", type=str, help="transformed data directory")

args = parser.parse_args()

train_data_path = args.input_data
train_data_file = os.path.join(train_data_path, 'train_data.csv')


with mlflow.start_run() as run:
    df = pd.read_csv(train_data_file)

    x = df.drop('Class', axis=1, inplace=False)
    y = df['Class']


    '''
    lr_cv = cross_validate(LogisticRegression(max_iter=400), x, y, cv=2, scoring=['balanced_accuracy', 'recall', 'precision'], return_train_score=True)
    print('Listo regresion lineal')
    #rf_cv = cross_validate(RandomForestClassifier(), x, y, cv=3, scoring=['balanced_accuracy', 'recall', 'precision'], return_train_score=True)
    #print('Listo random forest')
    svm = cross_validate(SVC(), x, y, cv = 3, scoring=['balanced_accuracy', 'recall', 'precision'], return_train_score=True)
    print('Listo SVM')

    hola =[]
    for score_model, name_model in [(lr_cv, 'lr'), (svm, 'svm')]:
        hola.append({
            f'train_acc_{name_model}': np.mean(score_model['train_balanced_accuracy']),
            f'train_recall_{name_model}': np.mean(score_model['train_recall']),
            f'train_precision_{name_model}': np.mean(score_model['train_precision']),
            f'test_acc_{name_model}': np.mean(score_model['test_balanced_accuracy']),
            f'test_recall_{name_model}': np.mean(score_model['test_recall']),
            f'test_precision_{name_model}': np.mean(score_model['test_precision'])
        })


    '''

    param_dist ={
        'penalty':['l1', 'l2', 'elasticnet'],
        'C': [c/10 for c in range(1,20,5)],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']

    }
    clf = RandomizedSearchCV(LogisticRegression(), param_dist, n_iter=5, random_state=0, refit='balanced_accuracy')
    '''
    mlflow.log_params({
        'penalty': 'l2',
        'C': 1.0,
        'solver':'lbfgs'
    })
    '''
    clf.fit(x,y)

    mlflow.sklearn.log_model(clf, artifact_path='skleran-model')
    mlflow.log_params(clf.best_params_)

    y_pred = clf.predict(x)

    recall = recall_score(y,y_pred)
    precision = precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    #print(recall)
    #print(precision)
    #print(acc)
    #print(conf_matrix)

    logging.info(f'matriz de confusion{conf_matrix}')

    
    mlflow.log_metrics({
        'recall':recall,
        'precision': precision,
        'accuracy': acc,

    })
    

#model_uri = f"runs:/{run.info.run_id}/sklearn-model"
#mv = mlflow.register_model(model_uri, "LogisticRegressionModel")
#logging.info("Name: {}".format(mv.name))
#logging.info("Version: {}".format(mv.version))

