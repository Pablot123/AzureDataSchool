import mlflow
import argparse



parser = argparse.ArgumentParser("moddel")
parser.add_argument('--model_out', type=str, help='training data')

args = parser.parse_args()


with mlflow.start_run() as run:
    # identificar el ultimo modelo que se sube con sus metricas
    runs = mlflow.search_runs(
        experiment_names=["exp-Airlines"]
    )

    #metricas actuales
    recent_run = runs[['run_id', 'metrics.accuracy validation', 'metrics.recall validation', 'metrics.precision validation', 'end_time' ]].loc[(runs['tags.mlflow.source.name'] == 'train_model.py') & (runs['status'] == 'FINISHED')].sort_values(by='end_time', ascending=False).head(1)
    recent_run_id = recent_run['run_id'].to_string().split()[1]
    

    run_info = mlflow.get_run(recent_run_id) 
    
    metrics = run_info.data.metrics
    
    actual_accuracy = round(metrics['accuracy validation'],5)
    actual_recall = metrics['precision validation']
    actual_precision = metrics['recall validation']

    #metricas del mejor modelo
    

    best_id_acc = runs[['run_id', 'metrics.accuracy validation']].loc[(runs['tags.mlflow.source.name'] == 'train_model.py') & (runs['status'] == 'FINISHED')].sort_values(by='metrics.accuracy validation', ascending=False).head(1)
    best_accuracy = round(best_id_acc['metrics.accuracy validation'].to_numpy()[0], 5)
    best_run_id = best_id_acc['run_id'].to_string().split()[1]
    
    
    # comparar con el mejor modelo
    if actual_accuracy > best_accuracy:
        #registrar el modelo
        print(f'ModelPath: {args.model_out}')
        model_name = 'airlines_model'
        #model = mlflow.sklearn.load_model(f"runs:/{recent_run_id}/sklearn-model")
        model = mlflow.sklearn.load_model(f'{args.model_out}/sklearn-model')

        #model_uri = f"runs:/{run.info.run_id}/sklearn-model"
        mlflow.sklearn.log_model(sk_model = model,
                                artifact_path = "sklearn-model",
                                registered_model_name=model_name)
    else:
        #se deja el mismo modelo
        print('no hay necesidad de pasarlo')


    # registrar el mejor modelo