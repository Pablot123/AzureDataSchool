import mlflow
from azureml.core import Workspace, Datastore, Run, Dataset
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from src.utils import dataset_transform
import argparse
import os


run = Run.get_context()

ws = run.experiment.workspace

#raw_data_test = run.input_datasets['raw_test_data'].to_pandas_dataframe()

parser = argparse.ArgumentParser("transform")
parser.add_argument('--input_data_test', type=str, help='test data')

args = parser.parse_args()

with mlflow.start_run():
    test_data_path = args.input_data_test
    test_data_file = os.path.join(test_data_path, 'transformed_data_test.csv')

    transformed_data_test = pd.read_csv(test_data_file)

    #transformed_data_test = dataset_transform(test_df=raw_data_test) #Colocar en un step diferente

    #load the latest version of the model
    model_name = 'airlines_model'
    model = mlflow.sklearn.load_model(f'models:/{model_name}/latest')
    print('loaded model')
    #mlflow.log_metric('prueba',1)

    pred = model.predict(transformed_data_test.drop('Class', axis=1, inplace=False))

    acc = balanced_accuracy_score(transformed_data_test['Class'], pred)
    mlflow.log_metric('acc', acc)

    pred_df = pd.DataFrame(pred, columns=['predicted_Class'])
    pred_df.to_csv('predicted_class_test.csv',index=False, header=True)
    mlflow.log_artifact('predicted_class_test.csv')




    



