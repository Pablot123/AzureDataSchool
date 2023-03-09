import mlflow

experiment_name = 'exp-prueba'
mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name="run-name") as run:
    #row_columns = len(df.columns)
    mlflow.log_metric('columnas', 1)
    