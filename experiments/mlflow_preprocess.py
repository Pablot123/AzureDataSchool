import mlflow
from scripts.myfuncs import cln_data, preprocs_data
import pandas as pd

from sklearn.model_selection import train_test_split

experiment_name ='exp-Airlines'
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    
    #loading data 
    #dataset = Dataset.get_by_name(ws, name='AirlinesDelay')
    df = pd.read_csv("../airlines_delay/airlines_delay.csv", sep = ",")

    df_clean= cln_data(df)

    train, test, = train_test_split(df_clean, test_size=0.3, random_state=73)
    
    train.to_csv('airlines_train.csv')
    test.to_csv('airlines_test.csv')

    mlflow.log_artifact('airlines_train.csv')
    mlflow.log_artifact('airlines_test.csv')

    x_train = train.drop('Class', axis=1, inplace=False)
    y_train = train['Class']
    
    preprocess_data_x = preprocs_data(x_train)

    preprocess_data_x.to_csv('airlines_train_processed.csv')
    mlflow.log_artifact('airlines_train_processed.csv')
