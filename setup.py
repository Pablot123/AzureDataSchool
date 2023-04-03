#Importing libraries
import azureml.core
from azureml.core import Workspace, Dataset
from azureml.core import Experiment
#from azureml.widgets import RunDetails
from azureml.core import  Environment
from azureml.core.compute import ComputeTarget, ComputeInstance
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
#from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import Pipeline
from azureml.pipeline.core import PipelineData
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep

import mlflow


'''
airlines_delay = '../airlines_delay/airlines_delay.csv'

ws = Workspace.from_config()

# Default datastore
default_store = ws.get_default_datastore() 

default_store.upload_files([airlines_delay], 
                           target_path = 'airlines', 
                           overwrite = True, 
                           show_progress = True)

print("Upload calls completed.")
'''

ws = Workspace.from_config()

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# nombre del cluster
compute_name = "prueba-DS"

# verificación de exixtencia del cluster
try:
    aml_compute = ComputeTarget(workspace=ws, name=compute_name)
    print('Existe!')
except ComputeTargetException:
    
    compute_config = ComputeInstance.provisioning_configuration(vm_size='Standard_DS11_v2',
                                                           ssh_public_access=False)
    aml_compute = ComputeTarget.create(ws, compute_name, compute_config)

aml_compute.wait_for_completion(show_output=True)


# Create a Python environment for the experiment (from a .yml file)
experiment_env = Environment.from_conda_specification("experiment_env",  './env/environment.yml')

# Register the environment 
experiment_env.register(workspace=ws)
registered_env = Environment.get(ws, 'experiment_env')

# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()

# Use the compute you created above. 
pipeline_run_config.target = aml_compute

# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env


#Linkg workspace with mlflow
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

airlines_data = ws.datasets.get('AirlinesDelay')

clean_airlines_data = OutputFileDatasetConfig('cleaned_data')

clean_step = PythonScriptStep(
    name="Clean airlines data",
    script_name="scripts/data_clean.py", 
    arguments=["--output_cleanse", clean_airlines_data],
    inputs=[airlines_data.as_named_input('raw_data')],
    outputs=[clean_airlines_data],
    compute_target=aml_compute,
    runconfig=pipeline_run_config,
    source_directory='./',
    allow_reuse=True
)

# train and test splits output
output_split_train = OutputFileDatasetConfig("output_split_train")
output_split_test = OutputFileDatasetConfig("output_split_test")
output_split_validation = OutputFileDatasetConfig("output_split_validation")

train_test_split_step = PythonScriptStep(
    name="split data train test",
    script_name="scripts/train_test_split.py", 
    arguments=["--input_data", clean_airlines_data.as_input(name='Clean_data'),
               "--output_train_data", output_split_train,
               "--output_test_data", output_split_test,
               "--output_val_data", output_split_validation],
    compute_target=aml_compute,
    runconfig=pipeline_run_config,
    source_directory='./',
    allow_reuse=True
)


transformed_data_train = OutputFileDatasetConfig('transformed_data_train')
transformed_data_val = OutputFileDatasetConfig('transformed_data_val')

transform_step = PythonScriptStep(
    name="transform airlines data",
    script_name="scripts/data_transform.py", 
    arguments=['--input_data_train', output_split_train.as_input(name='Train_data'),
               '--input_data_val',output_split_validation.as_input(name= 'Val_data'),
               '--input_data_test', None,
               "--output_transform_train", transformed_data_train,
               "--output_transform_val", transformed_data_val,
               "--output_transform_test", None],
    compute_target=aml_compute,
    runconfig=pipeline_run_config,
    source_directory='./',
    allow_reuse=True
)

datastore = ws.get_default_datastore()
step_output = PipelineData("model", datastore=datastore)

train_step = PythonScriptStep(
    name = 'Training model',
    script_name = 'scripts/train_model.py',
    arguments = ["--input_data_train", transformed_data_train.as_input(name='train_Data'),
                 "--input_data_val", transformed_data_val.as_input(name="val_data"),
                 "--output_model", step_output],
    #inputs = [output_split_train],
    outputs = [step_output],
    runconfig=pipeline_run_config,
    source_directory='./',
    allow_reuse=True

)

validation_step = PythonScriptStep(
    name = 'Validation model',
    script_name = 'scripts/val_model.py',
    arguments = ["--model_out", step_output.as_input(input_name='model_output')],
    inputs = [step_output],
    #outputs = [],
    runconfig=pipeline_run_config,
    source_directory='./',
    allow_reuse=True
)

print("Done.")

# Construct the pipeline
pipeline_steps = [clean_step, train_test_split_step, transform_step, train_step, validation_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
print("Pipeline is built.")


# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name = 'exp-Airlines')
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print("Pipeline submitted for execution.")
#RunDetails(pipeline_run).show()
pipeline_run.wait_for_completion(show_output=True)

#Linkg workspace with mlflow

ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

dataset_test = Dataset.get_by_name(ws, name='data_to_predict')
processed_data = OutputFileDatasetConfig('proccesed_data')

tracking_step = PythonScriptStep(
    name = 'traking',
    script_name="scripts_inference/tracking.py", 
    #arguments=["--output_data",processed_data],
    inputs=[dataset_test.as_named_input('raw_data')],
    compute_target=aml_compute,
    runconfig=pipeline_run_config,
    source_directory='./',
    allow_reuse=True
    )

preprocess_step = PythonScriptStep(
    name = 'preprocessing',
    script_name="scripts_inference/preprocess_data.py", 
    arguments=["--output_data",processed_data],
    inputs=[dataset_test.as_named_input('raw_data')],
    outputs=[processed_data],
    compute_target=aml_compute,
    runconfig=pipeline_run_config,
    source_directory='./',
    allow_reuse=True
)

inference_step = PythonScriptStep(
    name = 'inference',
    script_name = 'scripts_inference/inference.py',
    arguments = ['--input_data', processed_data.as_input(name='processed_data')],
    #inputs = [processed_data],
    compute_target=aml_compute,
    runconfig=pipeline_run_config,
    source_directory='./',
    allow_reuse=True
)

# Construct the pipeline

# Construct the pipeline
pipeline_steps_inference = [tracking_step, preprocess_step ,inference_step]
pipeline_inference = Pipeline(workspace=ws, steps=pipeline_steps_inference)
print("Pipeline is built.")


# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name = 'exp-Airlines_inference')
pipeline_run = experiment.submit(pipeline_inference, regenerate_outputs=True)
print("Pipeline submitted for execution.")
#RunDetails(pipeline_run).show()
pipeline_run.wait_for_completion(show_output=True)