import mlflow
from azureml.core import Workspace, Datastore, Run, Dataset
import pandas as pd
import argparse
import os

#parser = argparse.ArgumentParser("Inference")

#parser.add_argument("--output_preds", type=str, help="validation data directory")
#args = parser.parse_args()

ws = Run.get_context().experiment.workspace

with mlflow.start_run() as run:

    #load the latest version of the model
    model_name = 'airlines_model'
    model = mlflow.sklearn.load_model(f'models:/{model_name}/latest')
    print('loaded model')
    #mlflow.log_metric('prueba',1)

    #load the test dataset
    dataset = Dataset.get_by_name(ws, name='airlines_test_data', version='latest')
    test_df = dataset.to_pandas_dataframe()

    pred = model.predict(test_df.drop('Class', axis=1, inplace=False))

    #guardar en un archivo plano
    pred_df = pd.DataFrame(pred, columns=['predicted_Class'])
    #output_path = os.path.join(args.output_preds, 'test_predictions.csv')
    pred_df.to_csv('test_predictions.csv', index=False, header=True)
    mlflow.log_artifact('test_predictions.csv')

    



