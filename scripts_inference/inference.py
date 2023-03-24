import mlflow
from azureml.core import Workspace, Datastore, Run, Dataset
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from src.utils import dataset_transform
import argparse
import os


run = Run.get_context()

ws = run.experiment.workspace

raw_data_test = run.input_datasets['raw_test_data'].to_pandas_dataframe()

with mlflow.start_run():

    transformed_data_test = dataset_transform(test_df=raw_data_test)

    #load the latest version of the model
    model_name = 'airlines_model'
    model = mlflow.sklearn.load_model(f'models:/{model_name}/latest')
    print('loaded model')
    #mlflow.log_metric('prueba',1)

    pred = model.predict(transformed_data_test.drop('Class', axis=1, inplace=False))

    acc = balanced_accuracy_score(transformed_data_test['Class'], pred)
    mlflow.log_metric('acc', acc)

    #guardar en un archivo plano
    pred_df = pd.DataFrame(pred, columns=['predicted_Class'])
    #output_path = os.path.join(args.output_preds, 'test_predictions.csv')
    pred_df.to_csv('test_predictions.csv', index=False, header=True)
    mlflow.log_artifact('test_predictions.csv')



    



