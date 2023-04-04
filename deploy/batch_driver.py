import os
import mlflow
import pandas as pd
import numpy as np
from azureml.core import Run

def init():
    global model
    global preprocess
    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder
    # Please provide your model's folder name if there's one
    #model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")
    model_name = 'airlines_model'
    model = mlflow.sklearn.load_model(f'models:/{model_name}/latest')
    preprocess_name = 'scaler'
    preprocess = mlflow.sklearn.load_model(f'models:/{preprocess_name}/latest')


def run(mini_batch):
    #print(f"run method start: {__file__}, run({len(mini_batch)} files)")
    result_list = []
    result_list_path = []
    numeric_columns = ['Length','Time']
    category_columns = ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']
    
    for file_path in mini_batch:
        data = pd.read_csv(file_path)
        prueba = preprocess.transform(data.drop('Flight', axis=1, inplace=False))
        encoded_category = preprocess.named_transformers_['cat']['onehot'].get_feature_names_out(category_columns)
        labels = np.concatenate([numeric_columns, encoded_category])    
        prueba_df_all = pd.DataFrame(data=prueba, columns=labels)

        pred = model.predict(prueba_df_all)

        #resultList.append("{}: {}".format(os.path.basename(file_path), pred[0]))
        result_list_path.append(os.path.basename(file_path))
        result_list.append(pred[0])
        #df = pd.DataFrame(pred, columns=["predictions"])
        #df["file"] = os.path.basename(file_path)
        #resultList.extend(df.values)
        #  
    df = pd.DataFrame(data={'Path':result_list_path, 'Pred':result_list})
    df.to_csv('predictions.csv', header=True, index=False)

    with mlflow.start_run():
        ws = Run.get_context().experiment.workspace
        datastore = ws.get_default_datastore()

        train_data_reg = Dataset.Tabular.register_pandas_dataframe(df,
                                                                target=datastore,
                                                                name='predicted_data')

    
    return df